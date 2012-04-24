import vec;
import uint;

#[doc = "An IP address"]
enum ip_addr {
    /*
    Variant: ipv4

    An IPv4 address
    */
    ipv4(u8, u8, u8, u8),
}

#[doc = "Convert an `ip_addr` to a str"]
fn format_addr(ip: ip_addr) -> str {
    alt ip {
      ipv4(a, b, c, d) {
        #fmt["%u.%u.%u.%u", a as uint, b as uint, c as uint, d as uint]
      }
    }
}

#[doc = "
Convert a str to `ip_addr`

Converts a string of the format `x.x.x.x` into an ip_addr enum.

Fails if the string is not a valid IPv4 address
"]
fn parse_addr(ip: str) -> ip_addr {
    let parts = vec::map(str::split_char(ip, '.'), {|s|
        alt uint::from_str(s) {
          some(n) if n <= 255u { n }
          _ { fail "Invalid IP Address part." }
        }
    });
    if vec::len(parts) != 4u { fail "Too many dots in IP address"; }
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
