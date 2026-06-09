//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation

#[path = "../../utils/libc.rs"]
mod libc_utils;
#[path = "../../utils/mod.rs"]
mod utils;

use std::ffi::CString;
use std::ptr;

use libc_utils::*;

fn main() {
    test_getaddrinfo_freeaddrinfo();
}

/// Test doing address resolution using the `getaddrinfo` syscall.
/// This also tests freeing the address linked list using `freeaddrinfo`.
fn test_getaddrinfo_freeaddrinfo() {
    let node_c_str = CString::new("localhost").unwrap();
    let service_c_str = CString::new("8080").unwrap();

    let mut hints: libc::addrinfo = unsafe { std::mem::zeroed() };
    hints.ai_socktype = libc::SOCK_STREAM;
    let mut res: *mut libc::addrinfo = ptr::null_mut();
    unsafe {
        errno_check(libc::getaddrinfo(
            node_c_str.as_ptr(),
            service_c_str.as_ptr(),
            &hints,
            &mut res,
        ));
    }
    let start = res;
    let mut addr_count = 0;

    loop {
        unsafe {
            let Some(cur) = res.as_ref() else {
                // It's a null pointer so we're at the end of the linked list.
                break;
            };

            addr_count += 1;
            match (*cur).ai_family as libc::c_int {
                libc::AF_INET => {
                    let (_, addr) = net::sockname_ipv4(|storage, len| {
                        *(storage as *mut libc::sockaddr_in) = *cur.ai_addr.cast();
                        *len = (*res).ai_addrlen;
                        0
                    })
                    .unwrap();

                    let localhost_ipv4 = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 8080);
                    assert_eq!(localhost_ipv4.sin_family, addr.sin_family);
                    assert_eq!(localhost_ipv4.sin_port, addr.sin_port);
                    assert_eq!(localhost_ipv4.sin_addr.s_addr, addr.sin_addr.s_addr);
                }
                libc::AF_INET6 => {
                    let (_, addr) = net::sockname_ipv6(|storage, len| {
                        *(storage as *mut libc::sockaddr_in6) = *cur.ai_addr.cast();
                        *len = (*res).ai_addrlen;
                        0
                    })
                    .unwrap();

                    let localhost_ipv6 = net::sock_addr_ipv6(net::IPV6_LOCALHOST, 8080);
                    assert_eq!(localhost_ipv6.sin6_family, addr.sin6_family);
                    assert_eq!(localhost_ipv6.sin6_port, addr.sin6_port);
                    assert_eq!(localhost_ipv6.sin6_flowinfo, addr.sin6_flowinfo);
                    assert_eq!(localhost_ipv6.sin6_scope_id, addr.sin6_scope_id);
                    assert_eq!(localhost_ipv6.sin6_addr.s6_addr, addr.sin6_addr.s6_addr);
                }
                family => panic!("unexpected address family: {family}"),
            }

            res = cur.ai_next;
        }
    }

    // We expect an IPv4 and an IPv6 address.
    assert!(addr_count == 2);

    unsafe {
        libc::freeaddrinfo(start.cast());
    }
}
