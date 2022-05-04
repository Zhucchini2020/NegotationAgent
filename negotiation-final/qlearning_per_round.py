"""
CS1440/CS2440 Negotiation Final Project

reference: http://www.yasserm.com/scml/scml2020docs/tutorials.html
author: ninagawa

File to implement negotiation agent

"""

import warnings

from typing import Any
from typing import Dict
from typing import List

from negmas import Contract
from negmas import MechanismState
from negmas import NegotiatorMechanismInterface
from negmas import Outcome
from negmas import ResponseType
import numpy as np
from scml.oneshot import *

from print_helpers import *
from tier1_agent import LearningAgent, SimpleAgent

warnings.simplefilter("ignore")

learning_rate = 0.05
discount_factor = 0.90
exploration_rate = 0.05
num_states = 5
num_actions = 4
qTable = np.random.rand(num_states, num_actions)
trainingMode = True
turnToInitiate = 1

# qTable = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
# TODO: Change the class name to something unique. This will be the name of your agent.
class Primo(OneShotAgent):
    """
    My Agent

    Implement the methods for your agent in here!
    """

    def init(self):
        """
        Called once after the AWI (Agent World Interface) is set.

        Remarks:
            - Use this for any proactive initialization code.
        """
        self.qTable = qTable
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.discount_factor = discount_factor
        self.num_states = num_states
        self.num_actions = num_actions
        self.trainingMode = trainingMode
        self.mostRecentState = -1
        self.mostRecentUtility = -1
        self.mostRecentAction = -1
        self.turnToInitiate = turnToInitiate

    def before_step(self):
        """
        Called once every day before running the negotiations

        """
        # TODO
        self.secured = 0
        self._inv = self.ufun.invert()
        self._best_so_far, self._max_received = None, float("-inf")
        self._opp_min_amt_needed = float("inf")
        self.contracts = []
        self.ufun.find_limit(True)
        self.ufun.find_limit(False)
        self.action = -1
        self.opp_min_price = -1
        self.opp_max_price = -1

    def propose(self, negotiator_id: str, state: MechanismState) -> "Outcome":
        """
        Proposes an offer to one of the partners.

        Args:
            negotiator_id: ID of the negotiator (and partner)
            state: Mechanism state including current step

        Returns:
            an outcome to offer.
        """
        print(f"Agent {self.name} is proposing, step {state.step}")
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        if state.step <= self.turnToInitiate:
            # Begin training/data-gathering period
            offer[QUANTITY] = max(
                min(my_needs, self._opp_min_amt_needed),
                quantity_issue.min_value)
            offer[TIME] = self.awi.current_step
            if self._is_selling(ami):
                offer[UNIT_PRICE] = unit_price_issue.max_value
            else:
                offer[UNIT_PRICE] = unit_price_issue.min_value
            print(f"Because the step {state.step} is below {(3*ami.n_steps//4)}, propose a terrible price.")
            return tuple(offer)
        else:
            idle_util = self.ufun.from_contracts([])
            print(f"Because the step {state.step} is at or above {(3*ami.n_steps//4)}, propose the following.")
            offer[QUANTITY] = max(
                min(my_needs, self._opp_min_amt_needed),
                quantity_issue.min_value)
            if self.action == 0:
                print(f"Action 0: Continue offering the same price")
                offer[TIME] = self.awi.current_step
                if self._is_selling(ami):
                    offer[UNIT_PRICE] = unit_price_issue.max_value
                else:
                    offer[UNIT_PRICE] = unit_price_issue.min_value
                return tuple(offer)
            elif self.action == 1:
                offer[UNIT_PRICE] = self._find_good_price(ami, state, 0.2)
                print(f"Action 1: Offer price {offer[UNIT_PRICE]}")
                return tuple(offer)
            elif self.action == 2:
                offer[UNIT_PRICE] = self._find_good_price(ami, state, 1)
                print(f"Action 2: Offer price {offer[UNIT_PRICE]}")
                return tuple(offer)
            elif self.action == 3:
                offer[UNIT_PRICE] = self._find_good_price(ami, state, 5)
                print(f"Action 3: Offer price {offer[UNIT_PRICE]}")
                return tuple(offer)
        '''Old propose code:
        decay = self.ufun.find_limit(True).utility*3 #(1-state.relative_time)**0.5
        u = max(self.ufun.reserved_value, decay)
        ret_offer = self._best_so_far
        if ret_offer:
            if self._is_selling:
                ret_offer = (ret_offer[0], ret_offer[1], 1000000000)# ret_offer[2] + 1)
            else:
                ret_offer = (ret_offer[0], ret_offer[1], 0)# ret_offer[2] - 1)
        return self._best_so_far #if u < self._max_received else self._inv.one_in((u, u + 0.1), True)
        '''
    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        """
        Responds to an offer from one of the partners.

        Args:
            negotiator_id: ID of the negotiator (and partner)
            state: Mechanism state including current step
            offer: The offer received.

        Returns:
            A response type which can either be reject, accept, or end negotiation.

        Remarks:
            default behavior is to accept only if the current offer is the same
            or has a higher utility compared with what the agent would have
            proposed in the given state and reject otherwise

        """
        ami = self.get_nmi(negotiator_id)
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        amt_offered = offer[QUANTITY]
        self._opp_min_amt_needed = amt_offered
        priceRange = unit_price_issue.max_value - unit_price_issue.min_value

        if amt_offered > self._needed(negotiator_id):
            return ResponseType.REJECT_OFFER

        if state.step < 2:
            # Begin training/data-gathering period
            if self._is_selling(ami):
                if offer[UNIT_PRICE] < unit_price_issue.max_value:
                    return ResponseType.REJECT_OFFER
            else:
                if offer[UNIT_PRICE] > unit_price_issue.min_value:
                    return ResponseType.REJECT_OFFER
            return ResponseType.ACCEPT_OFFER
        elif state.step == 2:
            percentChanged = 0
            if self._is_selling(ami): 
                diffFromMin = offer[UNIT_PRICE] - unit_price_issue.min_value
                percentChanged = 1 - diffFromMin/priceRange 
                self.opp_min_price = offer[UNIT_PRICE]
            else:
                diffFromMax = offer[UNIT_PRICE] - unit_price_issue.min_value
                percentChanged = 1 - diffFromMax/priceRange
                self.opp_max_price = offer[UNIT_PRICE]
            currState = 0
            if percentChanged >= 0.99:
                currState = 0
            elif percentChanged >= 0.97:
                currState = 1
            elif percentChanged >= 0.95: 
                currState = 2
            elif percentChanged >= 0.90: 
                currState = 3
            else:
                currState = 4
            self.action = self.chooseNextMove(currState)
            if self.mostRecentState != -1 and self.mostRecentUtility != -1 and self.mostRecentAction != -1:
              
                self.updateRule(self.mostRecentAction, self.mostRecentState, currState, self.mostRecentUtility)
            self.mostRecentState = currState
            self.mostRecentAction = self.action
            return ResponseType.REJECT_OFFER
        else:
            idle_util = self.ufun.from_contracts(self.contracts)
            if self.action == 0:
                if state.step == ami.n_steps-1:
                    if self.ufun.from_offers(tuple([offer]), tuple([self._is_selling(ami)])) > idle_util:
                        print("accepted final offer")
                        return ResponseType.ACCEPT_OFFER
                    else:
                        print("rejected final offer")
                        return ResponseType.REJECT_OFFER
                else:
                    return ResponseType.REJECT_OFFER
            elif self.action == 1:
                if self._is_good_price(ami, state, offer[UNIT_PRICE],0.2):
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER
            elif self.action == 2:
                if self._is_good_price(ami, state, offer[UNIT_PRICE],1):
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER
            elif self.action == 3:
                if self._is_good_price(ami, state, offer[UNIT_PRICE],5):
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        """
        Called whenever a negotiation ends without agreement.

        Args:
            partners: List of the partner IDs consisting from self and the opponent.
            annotation: The annotation of the negotiation including the seller ID,
                        buyer ID, and the product.
            mechanism: The `NegotiatorMechanismInterface` instance containing all information
                       about the negotiation.
            state: The final state of the negotiation of the type `SAOState`
                   including the agreement if any.
        """
        # TODO
        print(f"Agent {self.name} failed to secure, trying action {self.action} up to step {state.step}")

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        """
        Called whenever a negotiation ends with agreement.

        Args:
            contract: The `Contract` agreed upon.
            mechanism: The `NegotiatorMechanismInterface` instance containing all information
                       about the negotiation that led to the `Contract` if any.
        """
        # TODO
        self.secured += contract.agreement["quantity"]
        self.contracts.append(contract)
        

    def step(self):
        """
        Called every step.

        Remarks:
            - Use this for any proactive code  that needs to be done every
              simulation step.
        """
        '''
        print(f"Utility for {self.name} is {self.ufun.from_contracts(self.contracts)}.")
        if self.ufun.ex_qout == 0:
             print(f"{self.name}'s exogenous contract had them buy {self.ufun.ex_qin} items at {self.ufun.ex_pin/self.ufun.ex_qin} each.")
        else:
            print(f"{self.name}'s exogenous contract had them sell {self.ufun.ex_qout} items at {self.ufun.ex_pout/self.ufun.ex_qout} each.")
        print(f"Shortfall penalty was {self.ufun.shortfall_penalty} and disposal cost was {self.ufun.disposal_cost}")
        print(f"Max possible utility was {self.ufun.max_utility} and min possible utility was {self.ufun.min_utility}")
        print(f"Utility of literally doing nothing would have been {self.ufun.from_contracts([])}")
        
        idleValue = self.ufun.from_contracts([])
        self.mostRecentUtility = self.ufun.from_contracts(self.contracts) - idleValue
        if len(self.contracts) == 0:
            print("No new contracts today")
        print(f"Tried action {self.action} today.")
        print(f"New Q Table for {self.name} is {self.qTable}")
        '''
        fileName = "qtab.txt"
        np.savetxt(fileName, self.qTable)



        
    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    def _needed(self, negotiator_id=None):
        return self.awi.current_exogenous_input_quantity + \
               self.awi.current_exogenous_output_quantity - \
               self.secured

    def updateRule(self, action, state, destState, reward):
        utility = self.ufun.from_contracts(self.contracts)
        maxDestState = np.max(self.qTable[destState])
        self.qTable[state][action] += self.learning_rate*(utility + self.discount_factor * maxDestState - self.qTable[state][action])
    
    def chooseNextMove(self, destState):
        if self.trainingMode:
            return np.random.randint(0,4)
        else:
            bestMove = np.argmax(self.qTable[destState])
            return bestMove
    
    def _find_good_price(self, ami, state, conc_exp):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(ami)
        th = self._th(state.step - self.turnToInitiate, ami.n_steps - self.turnToInitiate, conc_exp)
        # offer a price that is around th of your best possible price
        if self._is_selling(ami):
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)
    
    def _is_good_price(self, ami, state, price, conc_exp):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(ami)
        th = self._th(state.step - self.turnToInitiate, ami.n_steps - self.turnToInitiate, conc_exp)
        # a good price is one better than the threshold
        if self._is_selling(ami):
            return (price - mn) >= th * (mx - mn)
        else:
            return (mx - price) >= th * (mx - mn)

    def _price_range(self, ami):
        """Finds the minimum and maximum prices"""
        mn = ami.issues[UNIT_PRICE].min_value
        mx = ami.issues[UNIT_PRICE].max_value
        if self._is_selling(ami):
            mn = self.opp_min_price
            if mn == -1:
                raise ValueError("opp_min_price not updated correctly")
        else:
            mx = self.opp_max_price
            if mx == -1:
                raise ValueError("opp_max_price not updated correctly")
        return mn, mx

    def _th(self, step, n_steps, conc_exp):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1.0) / (n_steps - 1.0)) ** conc_exp



def main():
    """
    For more information:
    http://www.yasserm.com/scml/scml2020docs/tutorials/02.develop_agent_scml2020_oneshot.html
    """
    # TODO: Add/Remove agents from the list below to test your agent against other agents!
    #       (Make sure to change MyAgent to your agent class name)
    agents = [Primo, LearningAgent]
    world, ascores, tscores = try_agents(agents, n_trials = 1, draw=True) # change draw=True to see plot

    # TODO: Uncomment/Comment below to print/hide the individual agents' scores
    print_agent_scores(ascores)

    # TODO: Uncomment/Comment below to print/hide the average score of for each agent type
    print("Scores: ")
    print_type_scores(tscores)

    # TODO: Uncomment/Comment below to print/hide the exogenous contracts that drive the market
    print("Contracts:")
    print(analyze_contracts(world))


if __name__ == '__main__':
    main()