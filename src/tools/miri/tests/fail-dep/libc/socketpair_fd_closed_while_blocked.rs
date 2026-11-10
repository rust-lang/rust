//@ignore-target: windows # No libc socketpair on Windows
//@compile-flags: -Zmiri-fixed-schedule
//@error-in-other-file: deadlock
//@error-in-other-file: deadlock

use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

/// Test that `close` unblocks reads on the other FD.
fn main() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    let thread1 = thread::spawn(move || {
        // This read will be blocked because the buffer is already full.
        read_to_end(fds[0]).unwrap();
    });
    let thread2 = thread::spawn(move || {
        thread::yield_now(); // thread1 should go first
        // Close the FD thread1 is waiting on. This does NOT unblock thread1!
        // (That matches the behavior on native Linux.)
        errno_check(unsafe { libc::close(fds[0]) });
    });
    thread1.join().unwrap();
    thread2.join().unwrap();
}
