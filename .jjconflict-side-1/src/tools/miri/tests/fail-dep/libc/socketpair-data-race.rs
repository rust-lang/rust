//! This is a regression test for <https://github.com/rust-lang/miri/issues/3947>: we had some
//! faulty logic around `release_clock` that led to this code not reporting a data race.
//@ignore-target: windows # no libc socketpair on Windows
//@compile-flags: -Zmiri-deterministic-concurrency
use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::{errno_check, read_exact_array, write_all};

fn main() {
    static mut VAL: u8 = 0;
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });
    let thread1 = thread::spawn(move || {
        write_all(fds[0], b"a").unwrap();
        // The write to VAL is *after* the write to the socket, so there's no proper synchronization.
        unsafe { VAL = 1 };
    });
    thread::yield_now();

    let buf = read_exact_array::<1>(fds[1]).unwrap();
    assert_eq!(buf, "a".as_bytes());

    unsafe { assert_eq!({ VAL }, 1) }; //~ERROR: Data race

    thread1.join().unwrap();
}
