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
from sklearn.tree import export_graphviz

from print_helpers import *
from tier1_agent import LearningAgent, SimpleAgent

warnings.simplefilter("ignore")

learning_rate = 0.05
discount_factor = 0.90
exploration_rate = 0.05
num_states = 2*(10**2)
# states: buy/sell, exogenous price percentile, exogenous quant percentile, 
# oppent price percentile, quant percentile,time (5 each)
num_actions = 2
#qTable = np.random.rand(num_states, num_actions)
qTable = np.loadtxt("qtable.txt")
trainingMode = False

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
        #print(f"Agent {self.name} of type Primo initiated")

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
        self.currStates = []
        self.currResponses = []

    def propose(self, negotiator_id: str, state: MechanismState) -> "Outcome":
        """
        Proposes an offer to one of the partners.

        Args:
            negotiator_id: ID of the negotiator (and partner)
            state: Mechanism state including current step

        Returns:
            an outcome to offer.
        """
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        
        ami = self.get_nmi(negotiator_id)
        if not ami:
            return None
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]
        offer = [-1] * 3
        # Begin training/data-gathering period
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value),
            quantity_issue.min_value)
        offer[TIME] = self.awi.current_step
        if self._is_selling(ami):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)    
        
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
        
        opp_q = offer[QUANTITY]
        opp_p = offer[UNIT_PRICE]
        selling = 0
        ami = self.get_nmi(negotiator_id)
        mn = ami.issues[UNIT_PRICE].min_value
        mx = ami.issues[UNIT_PRICE].max_value
        ex_q = self.awi.current_exogenous_input_quantity + \
               self.awi.current_exogenous_output_quantity
        ex_p = self.awi.current_exogenous_input_price + \
               self.awi.current_exogenous_output_price
        if not ami:
            return None
        if self._is_selling(ami):
            selling = 1
        per_opp_q = self._percentile(mn,mx,opp_q)
        per_opp_p = self._percentile(mn,mx,opp_p)
        per_ex_q = self._percentile(mn,mx,ex_q)
        per_ex_p = self._percentile(mn,mx,ex_p)
        per_diff_p = per_opp_p-per_ex_p+5
        per_diff_q = per_opp_q-per_ex_q+5
        # time = self._percentile(0,ami.n_steps,state.step+1)
        #curr_state = 5**4 *(selling) + 5**3 *(per_opp_q) + 5**2 *(per_opp_p) + \
         #   5**1 *(per_ex_q) + 5**0 *(per_ex_p)
        curr_state = 10**2 * (selling) + 10*per_diff_p + per_diff_q
        self.currStates.append(curr_state)
        response = self.chooseNextMove(curr_state)
        self.currResponses.append(response)
        if response == 0:
            return ResponseType.REJECT_OFFER
        else:
            return ResponseType.ACCEPT_OFFER
        

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
        '''
        util= self.ufun.from_contracts(self.contracts)
        for i in range(len(self.currStates)):
            self.updateRule(self.currResponses[i], self.currStates[i], util)
        fileName = "qtable.txt"
        np.savetxt(fileName, self.qTable)
        #print(f"Agent {self.name} saved to the file.")

    def _percentile(self, min, max, num):
        percent = (num-min)/(max-min)
        if percent > 0.8:
            return 4
        elif percent > 0.6:
            return 3
        elif percent > 0.4:
            return 2
        elif percent > 0.2: 
            return 1
        else:
            return 0

        
    def _is_selling(self, ami):
        return ami.annotation["product"] == self.awi.my_output_product

    def _needed(self, negotiator_id=None):
        return self.awi.current_exogenous_input_quantity + \
               self.awi.current_exogenous_output_quantity - \
               self.secured

    def updateRule(self, action, state, utility):
        self.qTable[state][action] += self.learning_rate*(utility - self.qTable[state][action])
    
    def chooseNextMove(self, destState):
        if self.trainingMode:
            if np.random.rand() < self.exploration_rate:
                return np.random.randint(0,self.num_actions)
            else:
                bestMove = np.argmax(self.qTable[destState])
                return bestMove
        else:
            bestMove = np.argmax(self.qTable[destState])
            return bestMove
    
    def _find_good_price(self, ami, state, conc_exp):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(ami)
        th = self._th(state.step - (3*ami.n_steps//4), ami.n_steps - (3*ami.n_steps//4), conc_exp)
        # offer a price that is around th of your best possible price
        if self._is_selling(ami):
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)
    
    def _is_good_price(self, ami, state, price, conc_exp):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(ami)
        th = self._th(state.step - (3*ami.n_steps//4), ami.n_steps - (3*ami.n_steps//4), conc_exp)
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