//@only-target: linux android illumos
//@compile-flags: -Zmiri-deterministic-concurrency
//@error-in-other-file: deadlock

use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

// Test the behaviour of a thread being blocked on an eventfd read, get unblocked, and then
// get blocked again.

// The expected execution is
// 1. Thread 1 blocks.
// 2. Thread 2 blocks.
// 3. Thread 3 unblocks both thread 1 and thread 2.
// 4. Thread 1 reads.
// 5. Thread 2's `read` can never complete -> deadlocked.

fn main() {
    // eventfd write will block when EFD_NONBLOCK flag is clear
    // and the addition caused counter to exceed u64::MAX - 1.
    let flags = libc::EFD_CLOEXEC;
    let fd = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();

    let thread1 = thread::spawn(move || {
        // This read will block initially.
        let val = eventfd::read_val(fd).unwrap();
        assert_eq!(val, 1_u64);
    });

    let thread2 = thread::spawn(move || {
        // This read will block initially, then get unblocked by thread3, then get blocked again
        // because the `read` in thread1 executes first and set the counter to 0 again.
        let val = eventfd::read_val(fd).unwrap();
        //~^ERROR: deadlocked
        assert_eq!(val, 1_u64);
    });

    let thread3 = thread::spawn(move || {
        eventfd::write_val(fd, 1).unwrap();
    });

    thread1.join().unwrap();
    thread2.join().unwrap();
    thread3.join().unwrap();
}
