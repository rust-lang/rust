//@compile-flags: -Zmiri-deterministic-concurrency
//@only-target: linux android illumos
//@error-in-other-file: deadlock

use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::epoll::*;

// Using `as` cast since `EPOLLET` wraps around
const EPOLL_IN_OUT_ET: u32 = (EPOLLIN | EPOLLOUT | EPOLLET) as _;

// Test if only one thread is unblocked if multiple threads blocked on same epfd.
// Expected execution:
// 1. Thread 1 blocks.
// 2. Thread 2 blocks.
// 3. Thread 3 unblocks thread 2.
// 4. Thread 1 deadlocks.
fn main() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create an eventfd instance.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd1 = unsafe { libc::eventfd(0, flags) };
    // Make a duplicate so that we have two file descriptors for the same file description.
    let fd2 = unsafe { libc::dup(fd1) };

    // Register both with epoll.
    epoll_ctl_add(epfd, fd1, EPOLL_IN_OUT_ET as i32).unwrap();
    epoll_ctl_add(epfd, fd2, EPOLL_IN_OUT_ET as i32).unwrap();

    // Consume the initial events.
    check_epoll_wait::<8>(
        epfd,
        &[Ev { events: EPOLLOUT, data: fd1 }, Ev { events: EPOLLOUT, data: fd2 }],
        -1,
    );

    let thread1 = thread::spawn(move || {
        check_epoll_wait::<2>(
            epfd,
            &[Ev { events: EPOLLOUT, data: fd1 }, Ev { events: EPOLLOUT, data: fd2 }],
            -1,
        );
    });
    let thread2 = thread::spawn(move || {
        //~vERROR: deadlocked
        check_epoll_wait::<2>(
            epfd,
            &[Ev { events: EPOLLOUT, data: fd1 }, Ev { events: EPOLLOUT, data: fd2 }],
            -1,
        );
    });
    // Yield so the threads are both blocked.
    thread::yield_now();

    // Create two events at once.
    libc_utils::write_all(fd1, &0_u64.to_ne_bytes()).unwrap();

    thread1.join().unwrap();
    thread2.join().unwrap();
}
