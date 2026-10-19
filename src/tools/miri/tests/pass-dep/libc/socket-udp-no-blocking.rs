//@ignore-target: windows  # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation
//@run-native

#[path = "../../utils/libc.rs"]
mod libc_utils;

use libc_utils::*;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_send_not_connected();
    test_send_message_too_big();
}

/// Test sending bytes into a UDP socket with `send` but there hasn't been set a
/// default destination using the `connect` function. This should fail with EDESTADDRREQ.
fn test_send_not_connected() {
    let sockfd = unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };

    unsafe {
        // Change the socket to be non-blocking.
        errno_check(libc::fcntl(sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Attempt to write `TEST_BYTES` into the socket.
    let err = unsafe {
        errno_result(libc::send(sockfd, TEST_BYTES.as_ptr().cast(), TEST_BYTES.len(), 0))
            .unwrap_err()
    };
    // The `send` should fail since the socket has no default destination address set.
    assert_eq!(err.raw_os_error(), Some(libc::EDESTADDRREQ));
}

/// Test sending a UDP message which doesn't fit into a single UDP datagram.
/// The `send` operation should fail with EMSGSIZE.
fn test_send_message_too_big() {
    let (_rx_sockfd, addr) = net::make_local_ipv4(libc::SOCK_DGRAM).unwrap();
    let tx_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0)).unwrap() };

    unsafe {
        // Change the transmitting socket to be non-blocking.
        errno_check(libc::fcntl(tx_sockfd, libc::F_SETFL, libc::O_NONBLOCK));
    }

    // Set the default destination address of the transmitting socket.
    net::connect_ipv4(tx_sockfd, addr).unwrap();

    // The maximum UDP packet size is `u16::MAX` which is less than 70'000 bytes.
    let buffer = [1u8; 70_000];

    let err = unsafe {
        errno_result(libc::send(tx_sockfd, buffer.as_ptr().cast(), buffer.len(), 0)).unwrap_err()
    };
    // The `send` should fail with EMSGSIZE since the message doesn't fit into a UDP packet.
    assert_eq!(err.raw_os_error(), Some(libc::EMSGSIZE));
}
