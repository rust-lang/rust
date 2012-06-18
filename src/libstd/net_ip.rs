#[doc="
Types/fns concerning Internet Protocol (IP), versions 4 & 6
"];

import vec;
import uint;

import sockaddr_in = uv::ll::sockaddr_in;
import sockaddr_in6 = uv::ll::sockaddr_in6;
import uv_ip4_addr = uv::ll::ip4_addr;
import uv_ip4_name = uv::ll::ip4_name;

export ip_addr, parse_addr_err;
export format_addr;
export v4;

#[doc = "An IP address"]
enum ip_addr {
    #[doc="An IPv4 address"]
    ipv4(sockaddr_in),
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
      ipv4(addr) {
        unsafe {
            let result = uv_ip4_name(&addr);
            if result == "" {
                fail "failed to convert inner sockaddr_in address to str"
            }
            result
        }
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
          // FIXME: more copies brought to light to due the implicit
          // copy compiler warning.. what can be done? out pointers,
          // ala c#?
          result::ok(addr) { copy(addr) }
          result::err(err_data) {
            fail err_data.err_msg
          }
        }
    }
    fn try_parse_addr(ip: str) -> result::result<ip_addr,parse_addr_err> {
        unsafe {
            // need to figure out how to establish a parse failure..
            result::ok(ipv4(uv_ip4_addr(ip, 22)))
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_ipv4_parse_and_format_ip() {
        assert (format_addr(v4::parse_addr("127.0.0.1"))
                == "127.0.0.1")
    }
}