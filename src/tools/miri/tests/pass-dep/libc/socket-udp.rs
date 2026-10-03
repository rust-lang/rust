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
