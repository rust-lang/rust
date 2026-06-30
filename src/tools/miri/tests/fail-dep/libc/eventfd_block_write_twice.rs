//@only-target: linux android illumos
//@compile-flags: -Zmiri-deterministic-concurrency
//@error-in-other-file: deadlock

use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

// Test the behaviour of a thread being blocked on an eventfd `write`, get unblocked, and then
// get blocked again.

// The expected execution is
// 1. Thread 1 blocks.
// 2. Thread 2 blocks.
// 3. Thread 3 unblocks both thread 1 and thread 2.
// 4. Thread 1 writes u64::MAX.
// 5. Thread 2's `write` can never complete -> deadlocked.
fn main() {
    // eventfd write will block when EFD_NONBLOCK flag is clear
    // and the addition caused counter to exceed u64::MAX - 1.
    let flags = libc::EFD_CLOEXEC;
    let fd = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();
    // Write u64 - 1, so the all subsequent write will block.
    let sized_8_data: [u8; 8] = (u64::MAX - 1).to_ne_bytes();
    write_all(fd, &sized_8_data).unwrap();

    let thread1 = thread::spawn(move || {
        let sized_8_data = (u64::MAX - 1).to_ne_bytes();
        // Make sure that write is successful.
        write_all(fd, &sized_8_data).unwrap();
    });

    let thread2 = thread::spawn(move || {
        let sized_8_data = (u64::MAX - 1).to_ne_bytes();
        // Write u64::MAX - 1, so that all subsequent writes will block.
        let res = errno_result(unsafe {
            // This `write` will initially blocked, then get unblocked by thread3, then get blocked again
            // because the `write` in thread1 executes first.
            libc::write(fd, sized_8_data.as_ptr() as *const libc::c_void, 8)
            //~^ERROR: deadlocked
        })
        .unwrap();
        // Make sure that write is successful.
        assert_eq!(res, 8);
    });

    let thread3 = thread::spawn(move || {
        // This will unblock both `write` in thread1 and thread2.
        let buf: [u8; 8] = read_exact_array(fd).unwrap();
        let counter = u64::from_ne_bytes(buf);
        assert_eq!(counter, (u64::MAX - 1));
    });

    thread1.join().unwrap();
    thread2.join().unwrap();
    thread3.join().unwrap();
}
