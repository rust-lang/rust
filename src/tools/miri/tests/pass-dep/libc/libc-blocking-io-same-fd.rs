//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation

#[path = "../../utils/libc.rs"]
mod libc_utils;
use std::thread;
use std::time::Duration;

use libc_utils::*;

// This tests that the blocking I/O implementation works when multiple threads block on the
// same fd at the same time.

fn main() {
    let (server_sockfd, addr) = net::make_listener_ipv4(0).unwrap();
    let client_sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };

    // Spawn the server thread.
    let server_thread = thread::spawn(move || {
        let (peerfd, _) = net::accept_ipv4(server_sockfd).unwrap();

        // Yield back to reader threads to ensure that we have
        // two threads being blocked on the same fd at the same time.
        thread::sleep(Duration::from_millis(10));

        let mut buffer = [22; 128];
        let bytes_written = unsafe {
            errno_result(net::send_all(peerfd, buffer.as_mut_ptr().cast(), buffer.len(), 0))
                .unwrap()
        };
        assert_eq!(bytes_written as usize, 128);
    });

    net::connect_ipv4(client_sockfd, addr);

    let reader_thread = thread::spawn(move || {
        let mut buffer = [0; 8];
        let bytes_read = unsafe {
            errno_result(net::recv_all(client_sockfd, buffer.as_mut_ptr().cast(), buffer.len(), 0))
                .unwrap()
        };
        assert_eq!(bytes_read, 8);
    });

    let mut buffer = [0; 8];
    let bytes_read = unsafe {
        errno_result(net::recv_all(client_sockfd, buffer.as_mut_ptr().cast(), buffer.len(), 0))
            .unwrap()
    };
    assert_eq!(bytes_read, 8);

    reader_thread.join().unwrap();
    server_thread.join().unwrap();
}
