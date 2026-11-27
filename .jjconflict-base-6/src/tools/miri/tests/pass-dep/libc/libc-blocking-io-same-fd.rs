//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation -Zmiri-fixed-schedule

#[path = "../../utils/libc.rs"]
mod libc_utils;
use std::thread;

use libc_utils::*;

// This tests that the blocking I/O implementation works when multiple threads block on the
// same fd at the same time.

fn main() {
    let (server_sockfd, addr) = net::make_listener_ipv4().unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();

        // Yield back to reader threads to ensure that we have
        // two threads being blocked on the same fd at the same time.
        thread::yield_now();

        let mut buffer = [22u8; 128];
        unsafe {
            libc_utils::write_all_generic(
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                libc_utils::NoRetry,
                |buf, len| libc::send(peerfd, buf, len, 0),
            )
            .unwrap()
        };
    });

    net::connect_ipv4(client_sockfd, addr).unwrap();

    let reader_thread = thread::spawn(move || {
        let mut buffer = [0u8; 8];
        unsafe {
            libc_utils::read_exact_generic(
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                libc_utils::NoRetry,
                |buf, count| libc::recv(client_sockfd, buf, count, 0),
            )
            .unwrap()
        };
        assert_eq!(&buffer, &[22u8; 8]);
    });

    let mut buffer = [0u8; 8];
    unsafe {
        libc_utils::read_exact_generic(
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            libc_utils::NoRetry,
            |buf, count| libc::recv(client_sockfd, buf, count, 0),
        )
        .unwrap()
    };
    assert_eq!(&buffer, &[22u8; 8]);

    reader_thread.join().unwrap();
    server_thread.join().unwrap();
}
