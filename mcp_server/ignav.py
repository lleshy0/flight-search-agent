"""Ignav provider: the only file that knows Ignav's URLs and response structures.

Everything returned from here uses our own simple format, so server.py
(and Claude) never see Ignav-specific structures.
"""

import os
import math
import time
import logging
import httpx
import asyncio
from datetime import datetime
from typing import Literal, NotRequired, TypedDict

BASE_URL = "https://ignav.com/api"
API_KEY = os.environ.get("IGNAV_API_KEY")
MAX_ATTEMPTS = 3
logger = logging.getLogger(__name__)
DEFAULT_MARKET = "NL"  # sets the currency too: NL prices in EUR
MAX_RESULTS = 10  # ponytail: fixed cap to keep tool output small, add a `limit` param if the agent needs more

Cabin = Literal["economy", "premium_economy", "business", "first"]
SortBy = Literal["price", "duration"]


class TimeRange(TypedDict, total=False):
    """Airport-local hours, 0-23."""
    earliest_hour: int
    latest_hour: int
    arrival_earliest_hour: int
    arrival_latest_hour: int


class Leg(TypedDict):
    origin: str
    destination: str
    departure_date: str
    max_stops: NotRequired[int]
    departure_time_range: NotRequired[TimeRange]


async def _request(method: str, path: str, **kwargs) -> dict:
    """Send a request to Ignav and return the parsed JSON.
    Retries on HTTP 424 (temporary upstream failure)."""
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"X-Api-Key": API_KEY},
        timeout=30.0,
    ) as client:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.perf_counter()
            response = await client.request(method, path, **kwargs)
            # stderr via logging: stdout is the MCP stdio channel
            logger.info(
                "%s %s attempt %d: %d in %.1fs, request=%s",
                method, path, attempt, response.status_code,
                time.perf_counter() - started, kwargs.get("json") or kwargs.get("params"),
            )
            if response.status_code == 424 and attempt < MAX_ATTEMPTS:
                await asyncio.sleep(2 ** (attempt - 1))   # wait 1s, then 2s
                continue
            if response.is_error:
                # Surface Ignav's error body so the model can fix its arguments
                raise RuntimeError(f"Ignav {response.status_code}: {response.text}")
            return response.json()


def _clean(d: dict) -> dict:
    """Drop unset values: None, and the "", [] or {} some MCP clients send for blank optional
    fields, also inside nested dicts (time ranges, legs). 0 and False are real values and stay.
    Called on a function's locals() before any other local is assigned, this builds the
    request body, since our parameter names match Ignav's fields."""
    cleaned = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = _clean(v)
        elif isinstance(v, list):
            v = [_clean(x) if isinstance(x, dict) else x for x in v]
        if v is not None and v != "" and v != [] and v != {}:
            cleaned[k] = v
    return cleaned


def _format_duration(total_minutes: int | None) -> str | None:
    """'55m' under an hour, otherwise '2h 15m'."""
    if total_minutes is None:
        return None
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def _layover(arriving: dict, departing: dict) -> dict:
    """Connection between two consecutive segments. Both local times are at the same
    airport (or the same city, for an airport change), so they subtract directly."""
    wait = datetime.fromisoformat(departing["departure_time_local"]) - datetime.fromisoformat(arriving["arrival_time_local"])
    airport = arriving["arrival_airport"]
    if departing["departure_airport"] != airport:
        airport += f"→{departing['departure_airport']}"  # airport change during the layover
    return {"airport": airport, "duration": _format_duration(int(wait.total_seconds() // 60))}


def _simplify_leg(leg: dict) -> dict:
    segments = leg["segments"]
    return {
        "carrier": leg.get("carrier"),
        # Local times at the leg's first departure / final arrival airport
        "departs": segments[0]["departure_time_local"] if segments else None,
        "arrives": segments[-1]["arrival_time_local"] if segments else None,
        "duration": _format_duration(leg.get("duration_minutes")),
        "stops": max(len(segments) - 1, 0),
        "layovers": [_layover(a, b) for a, b in zip(segments, segments[1:])],
        "flights": [
            {
                "flight": f"{s['marketing_carrier_code'] or ''}{s['flight_number'] or ''}",
                "operated_by": s["operating_carrier_name"],
                "from": s["departure_airport"],
                "departs": s["departure_time_local"],
                "to": s["arrival_airport"],
                "arrives": s["arrival_time_local"],
            }
            for s in segments
        ],
    }


def _raw_legs(raw: dict) -> list[dict]:
    """Ignav's legs in travel order: a `legs` list for multi-city, else outbound (+ inbound)."""
    return raw.get("legs") or [raw["outbound"]] + ([raw["inbound"]] if raw.get("inbound") else [])


def _simplify_itinerary(raw: dict) -> dict:
    """Convert one Ignav itinerary (one-way, round-trip or multi-leg) into a compact form."""
    return {
        "id": raw.get("ignav_id"),
        "price": raw["price"]["amount"],
        "currency": raw["price"]["currency"],
        "price_verified": raw["price"]["status"] == "verified",
        "cabin_class": raw.get("cabin_class"),
        "bags": raw.get("bags"),
        "requires_self_transfer": raw.get("requires_self_transfer"),
        "legs": [_simplify_leg(leg) for leg in _raw_legs(raw)],
    }


def _total_minutes(raw: dict) -> float:
    return sum(leg.get("duration_minutes") or math.inf for leg in _raw_legs(raw))


def _simplify_results(data: dict, sort_by: SortBy) -> dict:
    # Sort Ignav's raw itineraries (numeric durations) before capping, so the cut drops the
    # worst matches. Each order breaks ties with the other; anything but "price" (incl. blank) is duration
    if sort_by == "price":
        key = lambda i: (i["price"]["amount"], _total_minutes(i))
    else:
        key = lambda i: (_total_minutes(i), i["price"]["amount"])
    itineraries = sorted(data["itineraries"], key=key)
    return {
        "observed_at": data.get("observed_at"),
        "total_found": len(itineraries),
        "itineraries": [_simplify_itinerary(i) for i in itineraries[:MAX_RESULTS]],
    }


async def search_airports(query: str, limit: int = 5) -> list[dict]:
    """Find IATA airport codes by city, airport name, country or code (query >= 2 chars).
    Use this to resolve places before searching flights."""
    return await _request("GET", "/airports", params={"q": query, "limit": limit})


async def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None = None,
    adults: int = 1,
    children: int = 0,
    cabin_class: Cabin = "economy",
    max_stops: int | None = None,
    max_price: int | None = None,
    min_carry_on_bags: int | None = None,
    min_checked_bags: int | None = None,
    departure_time_range: TimeRange | None = None,
    return_time_range: TimeRange | None = None,
    airlines_include: list[str] | None = None,
    airlines_exclude: list[str] | None = None,
    allow_self_transfer: bool = True,
    market: str = DEFAULT_MARKET,
    sort_by: SortBy = "duration",
) -> dict:
    """Search one-way fares, or round-trip fares when return_date is given.

    origin/destination: 3-letter IATA airport or metro code (e.g. SFO, LON).
    Dates: YYYY-MM-DD. max_stops: 0-2, omit for any. max_price: in the market's currency.
    airlines_include/exclude: 2-letter airline codes.
    market: 2-letter country code the search is priced and sold in; it also sets the currency.
    Default NL (prices in EUR); only change it if the user asks for another country or currency.
    Only set the filters the user's request calls for; leave the rest unset.
    sort_by: "duration" (default: shortest total travel time first, then cheapest) or
    "price" (cheapest first, then shortest).
    Returns the top 10 itineraries in that order, each with an id for get_booking_links.
    """
    body = {"market": DEFAULT_MARKET} | _clean(locals())  # a blank market falls back to ours, not Ignav's US
    body.pop("sort_by", None)
    # Check the cleaned body: a blank return_date ("") means one-way
    if "return_time_range" in body and "return_date" not in body:
        raise ValueError("return_time_range requires return_date")
    path = "/fares/round-trip" if "return_date" in body else "/fares/one-way"
    return _simplify_results(await _request("POST", path, json=body), sort_by)


async def search_multi_city(
    legs: list[Leg],
    adults: int = 1,
    children: int = 0,
    cabin_class: Cabin = "economy",
    max_price: int | None = None,
    min_carry_on_bags: int | None = None,
    min_checked_bags: int | None = None,
    airlines_include: list[str] | None = None,
    airlines_exclude: list[str] | None = None,
    allow_self_transfer: bool = True,
    market: str = DEFAULT_MARKET,
    sort_by: SortBy = "duration",
) -> dict:
    """Search a multi-city or open-jaw trip of 1-2 ordered legs priced as one journey.

    Each leg needs origin, destination, departure_date (YYYY-MM-DD, chronological) and may set
    max_stops and departure_time_range. Other filters and sort_by match search_flights.
    """
    body = {"market": DEFAULT_MARKET} | _clean(locals())  # a blank market falls back to ours, not Ignav's US
    body.pop("sort_by", None)
    return _simplify_results(await _request("POST", "/fares/search", json=body), sort_by)


async def search_cheapest_dates(
    origin: str,
    destination: str,
    start_date: str,
    end_date: str,
    stay_days: int | None = None,
    adults: int = 1,
    children: int = 0,
    cabin_class: Cabin = "economy",
    max_stops: int | None = None,
    max_price: int | None = None,
    min_checked_bags: int | None = None,
    departure_time_range: TimeRange | None = None,
    market: str = DEFAULT_MARKET,
) -> dict:
    """Lowest fare per departure date between start_date and end_date (inclusive, max 31 days).

    stay_days: omit for one-way, or 0-365 to price a return that many days later.
    Prices are indicative only; call search_flights on the chosen date to get bookable itineraries.
    """
    body = {"market": DEFAULT_MARKET} | _clean(locals())  # a blank market falls back to ours, not Ignav's US
    # pop with default: a blank date reaches Ignav as null and gets a readable validation error
    body["departure_date_range"] = {"start": body.pop("start_date", None), "end": body.pop("end_date", None)}
    data = await _request("POST", "/fares/cheapest-dates", json=body)
    dates = []
    for d in data["dates"]:
        price = d.get("price") or {}  # null when no fare found that day
        dates.append(_clean({
            "depart": d["departure_date"],
            "return": d.get("return_date"),
            "price": price.get("amount"),
            "currency": price.get("currency"),
        }))
    return {"dates": dates, "note": "Prices are estimates; confirm with search_flights."}


async def get_booking_links(itinerary_id: str, airline_only: bool = False) -> dict:
    """Get booking URLs for an itinerary id returned by search_flights or search_multi_city.
    Show URLs to the user verbatim. A link alone doesn't confirm seats or price.

    airline_only: keep only links to the airline's own website (no travel agencies).
    lands_on: "exact_flight" opens the chosen flight; "search_results" opens a search page
    where the user has to pick the flight again.
    """
    data = await _request("POST", "/fares/booking-links", json={"ignav_id": itinerary_id})
    options = []
    for opt in data["booking_options"]:
        links = []
        for link in opt["links"]:
            if airline_only and link.get("provider_type") != "airline":
                continue
            price = link.get("price") or {}
            links.append(_clean({
                "provider": link["provider_name"],
                "type": link.get("provider_type"),
                "lands_on": link.get("specificity"),
                "fare": link.get("fare_name"),
                "price": price.get("amount"),
                "currency": price.get("currency"),
                "url": link["url"],
            }))
        # One-way/round-trip options name their legs; multi-city options use leg indexes
        options.append({"legs": opt.get("legs") or opt.get("leg_indexes"), "links": links})
    price = data["itinerary"]["price"]
    note = "Pick one link from each option; together they cover the whole trip."
    if airline_only and any(not o["links"] for o in options):
        note += " Some options have no airline-direct link; call again without airline_only for other sellers."
    return {
        "total_price": price["amount"],
        "currency": price["currency"],
        "booking_options": options,
        "note": note,
    }
