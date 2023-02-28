from typing import List
from dataclasses import dataclass


@dataclass
class Order(object):
    CreatorID: int
    Side: bool
    Quantity: int
    Price: int


@dataclass
class Match(object):
    trade_name: str
    Bid: Order
    Offer: Order
    trading_price: float


class Market(object):
    def __init__(self):
        self.Bids: List[Order] = []
        self.Offers: List[Order] = []
        self.Matches: List[Match] = []

    def clear(self):
        self.Bids: List[Order] = []
        self.Offers: List[Order] = []
        self.Matches: List[Match] = []

    def AddOrder(self, order: Order):
        if order.Side:
            self.Offers.append(order)
        else:
            self.Bids.append(order)

    def MatchOrders(self):
        self.Bids = sorted(self.Bids, key=lambda x: x.Price)[::-1]
        self.Offers = sorted(self.Offers, key=lambda x: x.Price)

        while (len(self.Bids) > 0 and len(self.Offers) > 0):
            if self.Bids[0].Price < self.Offers[0].Price:
                break
            else:  # self.Bids[0].Price >= self.Offers[0].Price:
                currBid = self.Bids.pop(0)
                currOffer = self.Offers.pop(0)
                if currBid.Quantity != currOffer.Quantity:
                    if currBid.Quantity > currOffer.Quantity:
                        newBid = Order(currBid.CreatorID, currBid.Side, currBid.Quantity - currOffer.Quantity,
                                       currBid.Price)
                        self.Bids.insert(0, newBid)
                        currBid.Quantity = currOffer.Quantity
                    else:
                        newOffer = Order(currOffer.CreatorID, currOffer.Side, currOffer.Quantity - currBid.Quantity,
                                         currOffer.Price)
                        self.Offers.insert(0, newOffer)
                        currOffer.Quantity = currBid.Quantity
                self.Matches.append(Match('%d_to+%d' % (currOffer.CreatorID, currBid.CreatorID), currBid, currOffer,
                                          (currBid.Price + currOffer.Price) / 2))

    def ComputeClearingPrice(self) -> float:
        if len(self.Matches) == 0:
            return 0

        clearingPrice = 0
        cumulativeQuantity = 0
        for match in self.Matches:
            # print(match.Bid.CreatorID, match.Bid.Price, match.Bid.Quantity)
            # print(match.Offer.CreatorID, match.Offer.Price, match.Offer.Quantity)
            # print('trading_price: %f' % ((match.Bid.Price + match.Offer.Price) / 2))
            cumulativeQuantity += match.Bid.Quantity
            clearingPrice += match.Bid.Quantity * match.trading_price
        self.clear()
        # print('clearing_price:%f' % (clearingPrice / cumulativeQuantity))
        return clearingPrice / cumulativeQuantity
