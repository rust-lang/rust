#[doc="
Types/fns concerning Internet Protocol (IP), versions 4 & 6
"];

import vec;
import uint;

export ip_addr, parse_addr_err;
export format_addr;
export v4;

#[doc = "An IP address"]
enum ip_addr {
    #[doc="An IPv4 address"]
    ipv4(u8, u8, u8, u8),
    ipv6(u16,u16,u16,u16,u16,u16,u16,u16)
}

#[doc="
Human-friendly feedback on why a parse_addr attempt failed
"]
type parse_addr_err = {
    err_msg: str
};

#[doc="
Convert a `ip_addr` to a str

# Arguments

* ip - a `std::net::ip::ip_addr`
"]
fn format_addr(ip: ip_addr) -> str {
    alt ip {
      ipv4(a, b, c, d) {
        #fmt["%u.%u.%u.%u", a as uint, b as uint, c as uint, d as uint]
      }
      ipv6(_, _, _, _, _, _, _, _) {
        fail "FIXME (#2651) impl parsing of ipv6 addr";
      }
    }
}

mod v4 {
    #[doc = "
    Convert a str to `ip_addr`

    # Failure

j    Fails if the string is not a valid IPv4 address

    # Arguments

    * ip - a string of the format `x.x.x.x`

    # Returns

    * an `ip_addr` of the `ipv4` variant
    "]
    fn parse_addr(ip: str) -> ip_addr {
        alt try_parse_addr(ip) {
          result::ok(addr) { addr }
          result::err(err_data) {
            fail err_data.err_msg
          }
        }
    }
    fn try_parse_addr(ip: str) -> result::result<ip_addr,parse_addr_err> {
        let parts = vec::map(str::split_char(ip, '.'), {|s|
            alt uint::from_str(s) {
              some(n) if n <= 255u { n }
              _ { 256u }
            }
        });
        if vec::len(parts) != 4u {
            result::err({err_msg: #fmt("'%s' doesn't have 4 parts",
                        ip)})
        }
        else if vec::contains(parts, 256u) {
            result::err({err_msg: #fmt("invalid octal in provided addr '%s'",
                        ip)})
        }
        else {
            result::ok(ipv4(parts[0] as u8, parts[1] as u8,
                 parts[2] as u8, parts[3] as u8))
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_format_ip() {
        assert (format_addr(ipv4(127u8, 0u8, 0u8, 1u8))
                == "127.0.0.1")
    }

    #[test]
    fn test_parse_ip() {
        assert (v4::parse_addr("127.0.0.1") ==
                ipv4(127u8, 0u8, 0u8, 1u8));
    }
}