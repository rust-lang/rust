//@only-target: linux android illumos
//@compile-flags: -Zmiri-deterministic-concurrency
use std::thread;
use std::time::Duration;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

/// Test what happens when an eventfd gets closed while there is still a thread blocked on it.
fn main() {
    let fd = errno_result(unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) }).unwrap();

    // Spawn server thread.
    let server_thread = thread::spawn(move || {
        // Ensure main thread is blocked on reading from the client socket.
        thread::sleep(Duration::from_millis(10));

        unsafe { errno_check(libc::close(fd)) };
    });

    let val = eventfd::read_val(fd).unwrap(); //~ERROR: deadlock
    assert_eq!(val, 1);

    server_thread.join().unwrap();
}
