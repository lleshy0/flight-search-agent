from mcp.server.mcpserver import MCPServer

from mcp_server import ignav as provider

mcp = MCPServer("flight-search")

for tool in (
    provider.search_airports,
    provider.search_flights,
    provider.search_multi_city,
    provider.search_cheapest_dates,
    provider.get_booking_links,
):
    mcp.tool()(tool)

if __name__ == "__main__":
    mcp.run()
