//! This ensures that when an epoll_wait wakes up and there are multiple events,
//! and we only read one of them, we do not synchronize with the other events
//! and therefore still report a data race for things that need to see the second event
//! to be considered synchronized.
//@only-target: linux android illumos
// ensure deterministic schedule
//@compile-flags: -Zmiri-deterministic-concurrency

use std::thread;
use std::thread::spawn;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::epoll::*;
use libc_utils::*;

fn main() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create two socketpair instances.
    let mut fds_a = [-1, -1];
    unsafe {
        errno_check(libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds_a.as_mut_ptr()))
    };
    let mut fds_b = [-1, -1];
    unsafe {
        errno_check(libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds_b.as_mut_ptr()))
    };

    // Register both pipe read ends.
    epoll_ctl_add(epfd, fds_a[1], EPOLLIN | EPOLLET).unwrap();
    epoll_ctl_add(epfd, fds_b[1], EPOLLIN | EPOLLET).unwrap();

    static mut VAL_ONE: u8 = 40; // This one will be read soundly.
    static mut VAL_TWO: u8 = 50; // This one will be read unsoundly.
    let thread1 = spawn(move || {
        unsafe { VAL_ONE = 41 };

        let data = "abcde".as_bytes();
        libc_utils::write_all(fds_a[0], data).unwrap();

        unsafe { VAL_TWO = 51 };

        libc_utils::write_all(fds_b[0], data).unwrap();
    });
    thread::yield_now();

    // With room for one event: check result from epoll_wait.
    check_epoll_wait_partial(epfd, &[Ev { events: EPOLLIN, data: fds_a[1] }], 1, -1);

    // Since we only received one event, we have synchronized with
    // the write to VAL_ONE but not with the one to VAL_TWO.
    unsafe {
        assert_eq!({ VAL_ONE }, 41) // This one is not UB
    };
    unsafe {
        assert_eq!({ VAL_TWO }, 51) //~ERROR: Data race detected
    };

    thread1.join().unwrap();
}
