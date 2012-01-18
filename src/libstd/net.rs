/*
Module: net
*/

import vec;
import uint;

/* Section: Types */

/*
Tag: ip_addr

An IP address
*/
tag ip_addr {
    /*
    Variant: ipv4

    An IPv4 address
    */
    ipv4(u8, u8, u8, u8);
}

/* Section: Operations */

/*
Function: format_addr

Convert an <ip_addr> to a str
*/
fn format_addr(ip: ip_addr) -> str {
    alt ip {
      ipv4(a, b, c, d) {
        #fmt["%u.%u.%u.%u", a as uint, b as uint, c as uint, d as uint]
      }
      _ { fail "Unsupported address type"; }
    }
}

/*
Function: parse_addr

Convert a str to <ip_addr>

Converts a string of the format "x.x.x.x" into an ip_addr tag.

Failure:

String must be a valid IPv4 address
*/
fn parse_addr(ip: str) -> ip_addr {
    let parts = vec::map(str::split(ip, "."[0]), {|s| uint::from_str(s) });
    if vec::len(parts) != 4u { fail "Too many dots in IP address"; }
    for i in parts { if i > 255u { fail "Invalid IP Address part."; } }
    ipv4(parts[0] as u8, parts[1] as u8, parts[2] as u8, parts[3] as u8)
}

#[test]
fn test_format_ip() {
    assert (net::format_addr(net::ipv4(127u8, 0u8, 0u8, 1u8)) == "127.0.0.1")
}

#[test]
fn test_parse_ip() {
    assert (net::parse_addr("127.0.0.1") == net::ipv4(127u8, 0u8, 0u8, 1u8));
}
