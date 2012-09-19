//! Types/fns concerning Internet Protocol (IP), versions 4 & 6
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use iotask = uv::iotask::IoTask;
use interact = uv::iotask::interact;

use sockaddr_in = uv::ll::sockaddr_in;
use sockaddr_in6 = uv::ll::sockaddr_in6;
use addrinfo = uv::ll::addrinfo;
use uv_getaddrinfo_t = uv::ll::uv_getaddrinfo_t;
use uv_ip4_addr = uv::ll::ip4_addr;
use uv_ip4_name = uv::ll::ip4_name;
use uv_ip6_addr = uv::ll::ip6_addr;
use uv_ip6_name = uv::ll::ip6_name;
use uv_getaddrinfo = uv::ll::getaddrinfo;
use uv_freeaddrinfo = uv::ll::freeaddrinfo;
use create_uv_getaddrinfo_t = uv::ll::getaddrinfo_t;
use set_data_for_req = uv::ll::set_data_for_req;
use get_data_for_req = uv::ll::get_data_for_req;
use ll = uv::ll;
use comm = core::comm;

export IpAddr, parse_addr_err;
export format_addr;
export v4, v6;
export get_addr;
export Ipv4, Ipv6;

/// An IP address
enum IpAddr {
    /// An IPv4 address
    Ipv4(sockaddr_in),
    Ipv6(sockaddr_in6)
}

/// Human-friendly feedback on why a parse_addr attempt failed
type ParseAddrErr = {
    err_msg: ~str
};

/**
 * Convert a `ip_addr` to a str
 *
 * # Arguments
 *
 * * ip - a `std::net::ip::ip_addr`
 */
fn format_addr(ip: &IpAddr) -> ~str {
    match *ip {
      Ipv4(addr) =>  unsafe {
        let result = uv_ip4_name(&addr);
        if result == ~"" {
            fail ~"failed to convert inner sockaddr_in address to str"
        }
        result
      },
      Ipv6(addr) => unsafe {
        let result = uv_ip6_name(&addr);
        if result == ~"" {
            fail ~"failed to convert inner sockaddr_in address to str"
        }
        result
      }
    }
}

/// Represents errors returned from `net::ip::get_addr()`
enum IpGetAddrErr {
    GetAddrUnknownError
}

/**
 * Attempts name resolution on the provided `node` string
 *
 * # Arguments
 *
 * * `node` - a string representing some host address
 * * `iotask` - a `uv::iotask` used to interact with the underlying event loop
 *
 * # Returns
 *
 * A `result<~[ip_addr], ip_get_addr_err>` instance that will contain
 * a vector of `ip_addr` results, in the case of success, or an error
 * object in the case of failure
 */
fn get_addr(node: &str, iotask: iotask)
        -> result::Result<~[IpAddr], IpGetAddrErr> {
    do core::comm::listen |output_ch| {
        do str::as_buf(node) |node_ptr, len| unsafe {
            log(debug, fmt!("slice len %?", len));
            let handle = create_uv_getaddrinfo_t();
            let handle_ptr = ptr::addr_of(handle);
            let handle_data: GetAddrData = {
                output_ch: output_ch
            };
            let handle_data_ptr = ptr::addr_of(handle_data);
            do interact(iotask) |loop_ptr| unsafe {
                let result = uv_getaddrinfo(
                    loop_ptr,
                    handle_ptr,
                    get_addr_cb,
                    node_ptr,
                    ptr::null(),
                    ptr::null());
                match result {
                  0i32 => {
                    set_data_for_req(handle_ptr, handle_data_ptr);
                  }
                  _ => {
                    output_ch.send(result::Err(GetAddrUnknownError));
                  }
                }
            };
            output_ch.recv()
        }
    }
}

mod v4 {
    /**
     * Convert a str to `ip_addr`
     *
     * # Failure
     *
     * Fails if the string is not a valid IPv4 address
     *
     * # Arguments
     *
     * * ip - a string of the format `x.x.x.x`
     *
     * # Returns
     *
     * * an `ip_addr` of the `ipv4` variant
     */
    fn parse_addr(ip: &str) -> IpAddr {
        match try_parse_addr(ip) {
          result::Ok(addr) => copy(addr),
          result::Err(err_data) => fail err_data.err_msg
        }
    }
    // the simple, old style numberic representation of
    // ipv4
    type Ipv4Rep = { a: u8, b: u8, c: u8, d:u8 };

    trait AsUnsafeU32 {
        unsafe fn as_u32() -> u32;
    }

    impl Ipv4Rep: AsUnsafeU32 {
        // this is pretty dastardly, i know
        unsafe fn as_u32() -> u32 {
            *((ptr::addr_of(self)) as *u32)
        }
    }
    fn parse_to_ipv4_rep(ip: &str) -> result::Result<Ipv4Rep, ~str> {
        let parts = vec::map(str::split_char(ip, '.'), |s| {
            match uint::from_str(s) {
              Some(n) if n <= 255u => n,
              _ => 256u
            }
        });
        if vec::len(parts) != 4u {
                result::Err(fmt!("'%s' doesn't have 4 parts", ip))
                }
        else if vec::contains(parts, 256u) {
                result::Err(fmt!("invalid octal in addr '%s'", ip))
                }
        else {
            result::Ok({a: parts[0] as u8, b: parts[1] as u8,
                        c: parts[2] as u8, d: parts[3] as u8})
        }
    }
    fn try_parse_addr(ip: &str) -> result::Result<IpAddr,ParseAddrErr> {
        unsafe {
            let INADDR_NONE = ll::get_INADDR_NONE();
            let ip_rep_result = parse_to_ipv4_rep(ip);
            if result::is_err(ip_rep_result) {
                let err_str = result::get_err(ip_rep_result);
                return result::Err({err_msg: err_str})
            }
            // ipv4_rep.as_u32 is unsafe :/
            let input_is_inaddr_none =
                result::get(ip_rep_result).as_u32() == INADDR_NONE;

            let new_addr = uv_ip4_addr(str::from_slice(ip), 22);
            let reformatted_name = uv_ip4_name(&new_addr);
            log(debug, fmt!("try_parse_addr: input ip: %s reparsed ip: %s",
                            ip, reformatted_name));
            let ref_ip_rep_result = parse_to_ipv4_rep(reformatted_name);
            if result::is_err(ref_ip_rep_result) {
                let err_str = result::get_err(ref_ip_rep_result);
                return result::Err({err_msg: err_str})
            }
            if result::get(ref_ip_rep_result).as_u32() == INADDR_NONE &&
                 !input_is_inaddr_none {
                return result::Err(
                    {err_msg: ~"uv_ip4_name produced invalid result."})
            }
            else {
                result::Ok(Ipv4(copy(new_addr)))
            }
        }
    }
}
mod v6 {
    /**
     * Convert a str to `ip_addr`
     *
     * # Failure
     *
     * Fails if the string is not a valid IPv6 address
     *
     * # Arguments
     *
     * * ip - an ipv6 string. See RFC2460 for spec.
     *
     * # Returns
     *
     * * an `ip_addr` of the `ipv6` variant
     */
    fn parse_addr(ip: &str) -> IpAddr {
        match try_parse_addr(ip) {
          result::Ok(addr) => copy(addr),
          result::Err(err_data) => fail err_data.err_msg
        }
    }
    fn try_parse_addr(ip: &str) -> result::Result<IpAddr,ParseAddrErr> {
        unsafe {
            // need to figure out how to establish a parse failure..
            let new_addr = uv_ip6_addr(str::from_slice(ip), 22);
            let reparsed_name = uv_ip6_name(&new_addr);
            log(debug, fmt!("v6::try_parse_addr ip: '%s' reparsed '%s'",
                            ip, reparsed_name));
            // '::' appears to be uv_ip6_name() returns for bogus
            // parses..
            if  ip != &"::" && reparsed_name == ~"::" {
                result::Err({err_msg:fmt!("failed to parse '%s'",
                                           ip)})
            }
            else {
                result::Ok(Ipv6(new_addr))
            }
        }
    }
}

type GetAddrData = {
    output_ch: comm::Chan<result::Result<~[IpAddr],IpGetAddrErr>>
};

extern fn get_addr_cb(handle: *uv_getaddrinfo_t, status: libc::c_int,
                     res: *addrinfo) unsafe {
    log(debug, ~"in get_addr_cb");
    let handle_data = get_data_for_req(handle) as
        *GetAddrData;
    if status == 0i32 {
        if res != (ptr::null::<addrinfo>()) {
            let mut out_vec = ~[];
            log(debug, fmt!("initial addrinfo: %?", res));
            let mut curr_addr = res;
            loop {
                let new_ip_addr = if ll::is_ipv4_addrinfo(curr_addr) {
                    Ipv4(copy((
                        *ll::addrinfo_as_sockaddr_in(curr_addr))))
                }
                else if ll::is_ipv6_addrinfo(curr_addr) {
                    Ipv6(copy((
                        *ll::addrinfo_as_sockaddr_in6(curr_addr))))
                }
                else {
                    log(debug, ~"curr_addr is not of family AF_INET or "+
                        ~"AF_INET6. Error.");
                    (*handle_data).output_ch.send(
                        result::Err(GetAddrUnknownError));
                    break;
                };
                vec::push(out_vec, move new_ip_addr);

                let next_addr = ll::get_next_addrinfo(curr_addr);
                if next_addr == ptr::null::<addrinfo>() as *addrinfo {
                    log(debug, ~"null next_addr encountered. no mas");
                    break;
                }
                else {
                    curr_addr = next_addr;
                    log(debug, fmt!("next_addr addrinfo: %?", curr_addr));
                }
            }
            log(debug, fmt!("successful process addrinfo result, len: %?",
                            vec::len(out_vec)));
            (*handle_data).output_ch.send(result::Ok(move out_vec));
        }
        else {
            log(debug, ~"addrinfo pointer is NULL");
            (*handle_data).output_ch.send(
                result::Err(GetAddrUnknownError));
        }
    }
    else {
        log(debug, ~"status != 0 error in get_addr_cb");
        (*handle_data).output_ch.send(
            result::Err(GetAddrUnknownError));
    }
    if res != (ptr::null::<addrinfo>()) {
        uv_freeaddrinfo(res);
    }
    log(debug, ~"leaving get_addr_cb");
}

#[cfg(test)]
mod test {
    #[test]
    fn test_ip_ipv4_parse_and_format_ip() {
        let localhost_str = ~"127.0.0.1";
        assert (format_addr(&v4::parse_addr(localhost_str))
                == localhost_str)
    }
    #[test]
    fn test_ip_ipv6_parse_and_format_ip() {
        let localhost_str = ~"::1";
        let format_result = format_addr(&v6::parse_addr(localhost_str));
        log(debug, fmt!("results: expected: '%s' actual: '%s'",
            localhost_str, format_result));
        assert format_result == localhost_str;
    }
    #[test]
    fn test_ip_ipv4_bad_parse() {
        match v4::try_parse_addr(~"b4df00d") {
          result::Err(err_info) => {
            log(debug, fmt!("got error as expected %?", err_info));
            assert true;
          }
          result::Ok(addr) => {
            fail fmt!("Expected failure, but got addr %?", addr);
          }
        }
    }
    #[test]
    #[ignore(target_os="win32")]
    fn test_ip_ipv6_bad_parse() {
        match v6::try_parse_addr(~"::,~2234k;") {
          result::Err(err_info) => {
            log(debug, fmt!("got error as expected %?", err_info));
            assert true;
          }
          result::Ok(addr) => {
            fail fmt!("Expected failure, but got addr %?", addr);
          }
        }
    }
    #[test]
    #[ignore(reason = "valgrind says it's leaky")]
    fn test_ip_get_addr() {
        let localhost_name = ~"localhost";
        let iotask = uv::global_loop::get();
        let ga_result = get_addr(localhost_name, iotask);
        if result::is_err(ga_result) {
            fail ~"got err result from net::ip::get_addr();"
        }
        // note really sure how to realiably test/assert
        // this.. mostly just wanting to see it work, atm.
        let results = result::unwrap(ga_result);
        log(debug, fmt!("test_get_addr: Number of results for %s: %?",
                        localhost_name, vec::len(results)));
        for vec::each(results) |r| {
            let ipv_prefix = match *r {
              Ipv4(_) => ~"IPv4",
              Ipv6(_) => ~"IPv6"
            };
            log(debug, fmt!("test_get_addr: result %s: '%s'",
                            ipv_prefix, format_addr(r)));
        }
        // at least one result.. this is going to vary from system
        // to system, based on stuff like the contents of /etc/hosts
        assert vec::len(results) > 0;
    }
    #[test]
    #[ignore(reason = "valgrind says it's leaky")]
    fn test_ip_get_addr_bad_input() {
        let localhost_name = ~"sjkl234m,./sdf";
        let iotask = uv::global_loop::get();
        let ga_result = get_addr(localhost_name, iotask);
        assert result::is_err(ga_result);
    }
}
