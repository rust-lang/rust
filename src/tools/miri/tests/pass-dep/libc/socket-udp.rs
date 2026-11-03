//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation
//@run-native

#[path = "../../utils/libc.rs"]
mod libc_utils;
#[path = "../../utils/mod.rs"]
mod utils;

use libc_utils::*;

fn main() {
    test_getsockname_ipv4_connect();

    test_getpeername_ipv4();
    test_getpeername_ipv6();
}

/// Test the `getsockname` syscall on a IPv4 UDP socket for which the default
/// destination address has been set using `connect` but which has not been
/// explicitly bound to a local address.
/// The socket should be implicitly bound, resulting in a non-unspecified address
/// with a non-zero port being returned.
fn test_getsockname_ipv4_connect() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };

    // Destiniation address of the UDP socket.
    // There is no deeper meaning behind choice of the address and the port.
    let addr = net::sock_addr_ipv4([127, 0, 1, 3], 7564);

    net::connect_ipv4(sockfd, addr).unwrap();

    let (_, sock_addr) =
        net::sockname_ipv4(|storage, len| unsafe { libc::getsockname(sockfd, storage, len) })
            .unwrap();

    // We want to ensure that the local address is not the unspecified address.
    // Because the bound address could be of any local interface, we cannot
    // test for localhost.
    let addr = net::sock_addr_ipv4([0, 0, 0, 0], 0);

    assert_eq!(addr.sin_family, sock_addr.sin_family);
    assert_ne!(addr.sin_addr.s_addr, sock_addr.sin_addr.s_addr);
    assert!(sock_addr.sin_port > 0);
}

/// Test `getpeername` on an IPv4 socket.
/// For a socket whose default destination has been set using `connect`,
/// this should return the address of the specified default destination.
/// Otherwise, errno should be set to ENOTCONN.
fn test_getpeername_ipv4() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };

    // When no default destination has been set using `connect`,
    // `getpeername` should fail and set errno to ENOTCONN.
    let err = net::sockname_ipv4(|storage, len| unsafe { libc::getpeername(sockfd, storage, len) })
        .unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::ENOTCONN));

    let addr = net::sock_addr_ipv4(net::IPV4_LOCALHOST, 1234);
    // Explicitly set default destination address.
    net::connect_ipv4(sockfd, addr).unwrap();

    let (_, dest_addr) =
        net::sockname_ipv4(|storage, len| unsafe { libc::getpeername(sockfd, storage, len) })
            .unwrap();

    assert_eq!(addr.sin_family, dest_addr.sin_family);
    assert_eq!(addr.sin_port, dest_addr.sin_port);
    assert_eq!(addr.sin_addr.s_addr, dest_addr.sin_addr.s_addr);
}

/// Test `getpeername` on an IPv6 socket.
/// For a socket whose default destination has been set using `connect`,
/// this should return the address of the specified default destination.
/// Otherwise, errno should be set to ENOTCONN.
fn test_getpeername_ipv6() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET6, libc::SOCK_DGRAM, 0)).unwrap() };

    // When no default destination has been set using `connect`,
    // `getpeername` should fail and set errno to ENOTCONN.
    let err = net::sockname_ipv6(|storage, len| unsafe { libc::getpeername(sockfd, storage, len) })
        .unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::ENOTCONN));

    let addr = net::sock_addr_ipv6(net::IPV6_LOCALHOST, 1234);
    // Explicitly set default destination address.
    net::connect_ipv6(sockfd, addr).unwrap();

    let (_, dest_addr) =
        net::sockname_ipv6(|storage, len| unsafe { libc::getpeername(sockfd, storage, len) })
            .unwrap();

    assert_eq!(addr.sin6_family, dest_addr.sin6_family);
    assert_eq!(addr.sin6_port, dest_addr.sin6_port);
    assert_eq!(addr.sin6_flowinfo, dest_addr.sin6_flowinfo);
    assert_eq!(addr.sin6_scope_id, dest_addr.sin6_scope_id);
    assert_eq!(addr.sin6_addr.s6_addr, dest_addr.sin6_addr.s6_addr);
}
