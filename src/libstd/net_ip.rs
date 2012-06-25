#[doc="
Types/fns concerning Internet Protocol (IP), versions 4 & 6
"];

import vec;
import uint;
import iotask = uv::iotask::iotask;
import interact = uv::iotask::interact;
import comm::methods;

import sockaddr_in = uv::ll::sockaddr_in;
import sockaddr_in6 = uv::ll::sockaddr_in6;
import addrinfo = uv::ll::addrinfo;
import uv_getaddrinfo_t = uv::ll::uv_getaddrinfo_t;
import uv_ip4_addr = uv::ll::ip4_addr;
import uv_ip4_name = uv::ll::ip4_name;
import uv_ip6_addr = uv::ll::ip6_addr;
import uv_ip6_name = uv::ll::ip6_name;
import uv_getaddrinfo = uv::ll::getaddrinfo;
import uv_freeaddrinfo = uv::ll::freeaddrinfo;
import create_uv_getaddrinfo_t = uv::ll::getaddrinfo_t;
import set_data_for_req = uv::ll::set_data_for_req;
import get_data_for_req = uv::ll::get_data_for_req;
import ll = uv::ll;

export ip_addr, parse_addr_err;
export format_addr;
export v4, v6;
export get_addr;

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

type get_addr_data = {
    output_ch: comm::chan<result::result<[ip_addr],ip_get_addr_err>>
};

crust fn get_addr_cb(handle: *uv_getaddrinfo_t, status: libc::c_int,
                     res: *addrinfo) unsafe {
    log(debug, "in get_addr_cb");
    let handle_data = get_data_for_req(handle) as
        *get_addr_data;
    if status == 0i32 {
        if res != (ptr::null::<addrinfo>()) {
            let mut out_vec = [];
            let mut addr_strings = [];
            log(debug, #fmt("initial addrinfo: %?", res));
            let mut curr_addr = res;
            loop {
                let new_ip_addr = if ll::is_ipv4_addrinfo(curr_addr) {
                    ipv4(copy((
                        *ll::addrinfo_as_sockaddr_in(curr_addr))))
                }
                else {
                    ipv6(copy((
                        *ll::addrinfo_as_sockaddr_in6(curr_addr))))
                };
                // we're doing this check to avoid adding multiple
                // ip_addrs to the out_vec that are duplicates.. on
                // 64bit unbuntu a call to uv_getaddrinfo against
                // localhost seems to return three addrinfos, all
                // distinct (by ptr addr), that are all ipv4
                // addresses and all point to 127.0.0.1
                let addr_str = format_addr(new_ip_addr);
                if !vec::contains(addr_strings, addr_str) {
                    addr_strings += [addr_str];
                    out_vec += [new_ip_addr];
                }

                let next_addr = ll::get_next_addrinfo(curr_addr);
                if next_addr == ptr::null::<addrinfo>() as *addrinfo {
                    log(debug, "null next_addr encountered. no mas");
                    break;
                }
                else {
                    curr_addr = next_addr;
                    log(debug, #fmt("next_addr addrinfo: %?", curr_addr));
                }
            }
            log(debug, #fmt("successful process addrinfo result, len: %?",
                            vec::len(out_vec)));
            (*handle_data).output_ch.send(result::ok(out_vec));
        }
        else {
            log(debug, "addrinfo pointer is NULL");
            (*handle_data).output_ch.send(
                result::err(get_addr_unknown_error));
        }
    }
    else {
        log(debug, "status != 0 error in get_addr_cb");
        (*handle_data).output_ch.send(
            result::err(get_addr_unknown_error));
    }
    if res != (ptr::null::<addrinfo>()) {
        uv_freeaddrinfo(res);
    }
    log(debug, "leaving get_addr_cb");
}

#[doc="
"]
enum ip_get_addr_err {
    get_addr_unknown_error
}

#[doc="
"]
fn get_addr(++node: str, iotask: iotask)
        -> result::result<[ip_addr], ip_get_addr_err> unsafe {
    comm::listen {|output_ch|
        str::unpack_slice(node) {|node_ptr, len|
            log(debug, #fmt("slice len %?", len));
            let handle = create_uv_getaddrinfo_t();
            let handle_ptr = ptr::addr_of(handle);
            let handle_data: get_addr_data = {
                output_ch: output_ch
            };
            let handle_data_ptr = ptr::addr_of(handle_data);
            interact(iotask) {|loop_ptr|
                let result = uv_getaddrinfo(
                    loop_ptr,
                    handle_ptr,
                    get_addr_cb,
                    node_ptr,
                    ptr::null(),
                    ptr::null());
                alt result {
                  0i32 {
                    set_data_for_req(handle_ptr, handle_data_ptr);
                  }
                  _ {
                    output_ch.send(result::err(get_addr_unknown_error));
                  }
                }
            };
            output_ch.recv()
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
                result::ok(ipv4(copy(new_addr)))
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

#[cfg(test)]
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
    #[test]
    fn test_get_addr() {
        let localhost_name = "localhost";
        let iotask = uv::global_loop::get();
        let ga_result = get_addr(localhost_name, iotask);
        if result::is_err(ga_result) {
            fail "got err result from net::ip::get_addr();"
        }
        // note really sure how to realiably test/assert
        // this.. mostly just wanting to see it work, atm.
        let results = result::unwrap(ga_result);
        log(debug, #fmt("test_get_addr: Number of results for %s: %?",
                        localhost_name, vec::len(results)));
        for vec::each(results) {|r|
            let ipv_prefix = alt r {
              ipv4(_) {
                "IPv4"
              }
              ipv6(_) {
                "IPv6"
              }
            };
            log(debug, #fmt("test_get_addr: result %s: '%s'",
                            ipv_prefix, format_addr(r)));
        }
    }
}