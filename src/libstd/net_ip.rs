#[doc="
Types/fns concerning Internet Protocol (IP), versions 4 & 6
"];

import vec;
import uint;

import sockaddr_in = uv::ll::sockaddr_in;
import sockaddr_in6 = uv::ll::sockaddr_in6;
import uv_ip4_addr = uv::ll::ip4_addr;
import uv_ip4_name = uv::ll::ip4_name;
import uv_ip6_addr = uv::ll::ip6_addr;
import uv_ip6_name = uv::ll::ip6_name;

export ip_addr, parse_addr_err;
export format_addr;
export v4;

#[doc = "An IP address"]
enum ip_addr {
    #[doc="An IPv4 address"]
    ipv4(sockaddr_in),
    ipv6(sockaddr_in6)
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
      ipv6(addr) {
        unsafe {
            let result = uv_ip6_name(&addr);
            if result == "" {
                fail "failed to convert inner sockaddr_in address to str"
            }
            result
        }
      }
    }
}

mod v4 {
    #[doc = "
    Convert a str to `ip_addr`

    # Failure

    Fails if the string is not a valid IPv4 address

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
            let new_addr = uv_ip4_addr(ip, 22);
            let reformatted_name = uv_ip4_name(&new_addr);
            log(debug, #fmt("try_parse_addr: input ip: %s reparsed ip: %s",
                            ip, reformatted_name));
            // here we're going to
            let inaddr_none_val = "255.255.255.255";
            if ip != inaddr_none_val && reformatted_name == inaddr_none_val {
                result::err({err_msg:#fmt("failed to parse '%s'",
                                           ip)})
            }
            else {
                result::ok(ipv4(new_addr))
            }
        }
    }
}
mod v6 {
    #[doc = "
    Convert a str to `ip_addr`

    # Failure

    Fails if the string is not a valid IPv6 address

    # Arguments

    * ip - an ipv6 string. See RFC2460 for spec.

    # Returns

    * an `ip_addr` of the `ipv6` variant
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
            let new_addr = uv_ip6_addr(ip, 22);
            let reparsed_name = uv_ip6_name(&new_addr);
            log(debug, #fmt("v6::try_parse_addr ip: '%s' reparsed '%s'",
                            ip, reparsed_name));
            // '::' appears to be uv_ip6_name() returns for bogus
            // parses..
            if  ip != "::" && reparsed_name == "::" {
                result::err({err_msg:#fmt("failed to parse '%s'",
                                           ip)})
            }
            else {
                result::ok(ipv6(new_addr))
            }
        }
    }
}

//#[cfg(test)]
mod test {
    #[test]
    fn test_ipv4_parse_and_format_ip() {
        let localhost_str = "127.0.0.1";
        assert (format_addr(v4::parse_addr(localhost_str))
                == localhost_str)
    }
    #[test]
    fn test_ipv6_parse_and_format_ip() {
        let localhost_str = "::1";
        let format_result = format_addr(v6::parse_addr(localhost_str));
        log(debug, #fmt("results: expected: '%s' actual: '%s'",
            localhost_str, format_result));
        assert format_result == localhost_str;
    }
    #[test]
    fn test_ipv4_bad_parse() {
        alt v4::try_parse_addr("b4df00d") {
          result::err(err_info) {
            log(debug, #fmt("got error as expected %?", err_info));
            assert true;
          }
          result::ok(addr) {
            fail #fmt("Expected failure, but got addr %?", addr);
          }
        }
    }
    #[test]
    fn test_ipv6_bad_parse() {
        alt v6::try_parse_addr("::,~2234k;") {
          result::err(err_info) {
            log(debug, #fmt("got error as expected %?", err_info));
            assert true;
          }
          result::ok(addr) {
            fail #fmt("Expected failure, but got addr %?", addr);
          }
        }
    }
}