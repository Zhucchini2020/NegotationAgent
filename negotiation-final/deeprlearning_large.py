"""
CS1440/CS2440 Negotiation Final Project

reference: http://www.yasserm.com/scml/scml2020docs/tutorials.html
author: ninagawa

File to implement negotiation agent

"""

from multiprocessing import reduction
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import exists

warnings.simplefilter("ignore")

learning_rate = 0.001
propose_file_name = 'propose_network_qlearning_big4.pt'
respond_file_name = 'respond_network_qlearning_big4.pt'
finetune = True
# TODO: Change the class name to something unique. This will be the name of your agent.

class ProposeNet(nn.Module):
    def __init__(self):
        super(ProposeNet, self).__init__()

        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 1000) 
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc5(x))
        return x

class RespondNet(nn.Module):
    def __init__(self):
        super(RespondNet, self).__init__()

        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 256)
        self.fc5 = nn.Linear(256, 1) 
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x

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
        
        if exists(propose_file_name):
            self.propose_model = torch.load(propose_file_name)
        else:
            self.propose_model = ProposeNet()
        if exists(respond_file_name):
            self.respond_model = torch.load(respond_file_name)
        else:
            self.respond_model = RespondNet()
        self.propose_optimizer = torch.optim.RMSprop(self.propose_model.parameters(), lr=learning_rate)
        self.respond_optimizer = torch.optim.RMSprop(self.respond_model.parameters(), lr=learning_rate)

    def before_step(self):
        """
        Called once every day before running the negotiations

        """
        # TODO
        self.secured = 0
        self.contracts = []
        self.preds = []
        self.proposes = []
        self.offers = {}


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
        if negotiator_id not in self.offers:
            return None
        prev_offer = self.offers[negotiator_id]
        quantity = prev_offer[QUANTITY]
        time = prev_offer[TIME]
        price = prev_offer[UNIT_PRICE]
        selling = self._is_selling(ami)
        offer = [-1] * 3
        input = torch.tensor([quantity, time, price, selling]).type(torch.FloatTensor)
        output = self.propose_model(input)
        probs = torch.distributions.Categorical(output)
        a = probs.sample().item()
        offer[QUANTITY] = a % 100
        offer[TIME] = self.awi.current_step
        offer[UNIT_PRICE] = a % 10
        offer = tuple(offer)
        util = self.ufun.from_offers(tuple([offer]), tuple([self._is_selling(ami)]))
        self.proposes.append((input, a, util))
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
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        self.offers[negotiator_id] = offer
        ami = self.get_nmi(negotiator_id)
        opp_q = offer[QUANTITY]
        opp_p = offer[UNIT_PRICE]
        selling = self._is_selling(ami)
        th = self._th(state.step - (3*ami.n_steps//4), ami.n_steps - (3*ami.n_steps//4), 0.2)
        input = torch.tensor([opp_p, opp_q, int(selling), th]).type(torch.FloatTensor)
        output = self.respond_model(input)
        probs = torch.distributions.Bernoulli(output)
        a = probs.sample()
        response = ResponseType.ACCEPT_OFFER if a == 1 else ResponseType.REJECT_OFFER
        util = self.ufun.from_offers(tuple([offer]), tuple([self._is_selling(ami)]))
        self.preds.append((input, a, util))
        return response
        

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
        '''avg_util = self.ufun.from_contracts(self.contracts)
        for pred, contract in zip(self.preds, self.contracts):
            label = torch.tensor(1) if self.ufun.from_contracts([contract]) > avg_util else torch.tensor(0)
            loss = self.loss_fn(pred, label)
            self.model.zero_grad()
            loss.backward()
        with torch.no_grad():
            for param in self.model.parameters():
                param -= learning_rate * param.grad'''
        self.propose_optimizer.zero_grad()
        self.respond_optimizer.zero_grad()
        avg_util = self.ufun.from_contracts(self.contracts)
        max_util = torch.tensor(self.ufun.find_limit(True).utility)
        for state, action, reward in self.proposes:
            #reward -= avg_util
            output = self.propose_model(state)
            probs = torch.distributions.Categorical(output)
            mul = max_util - reward
            loss = -probs.log_prob(torch.autograd.Variable(torch.tensor(action))) * mul
            loss.backward()
        self.propose_optimizer.step()
        for state, action, reward in self.preds:
            #reward -= avg_util
            output = self.respond_model(state)
            probs = torch.distributions.Bernoulli(output)
            mul = max_util - reward
            loss = -probs.log_prob(torch.autograd.Variable(torch.tensor(action))) * mul
            loss.backward()
        self.respond_optimizer.step()
        if not finetune:
            torch.save(self.respond_model, respond_file_name)
            torch.save(self.propose_model, propose_file_name)


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
    for i in range(10):
        world, ascores, tscores = try_agents(agents, n_trials = 1, draw=True) # change draw=True to see plot

    # TODO: Uncomment/Comment below to print/hide the individual agents' scores
    # print_agent_scores(ascores)

    # TODO: Uncomment/Comment below to print/hide the average score of for each agent type
        print("Scores: ")
        print_type_scores(tscores)
    
    # TODO: Uncomment/Comment below to print/hide the exogenous contracts that drive the market
    #print("Contracts:")
    #print(analyze_contracts(world))


if __name__ == '__main__':
    main()