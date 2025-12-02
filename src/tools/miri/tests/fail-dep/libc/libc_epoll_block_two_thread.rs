//@compile-flags: -Zmiri-deterministic-concurrency
//@only-target: linux android illumos
//@error-in-other-file: deadlock

use std::convert::TryInto;
use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;

// Using `as` cast since `EPOLLET` wraps around
const EPOLL_IN_OUT_ET: u32 = (libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET) as _;

#[track_caller]
fn check_epoll_wait<const N: usize>(
    epfd: i32,
    expected_notifications: &[(u32, u64)],
    timeout: i32,
) {
    let epoll_event = libc::epoll_event { events: 0, u64: 0 };
    let mut array: [libc::epoll_event; N] = [epoll_event; N];
    let maxsize = N;
    let array_ptr = array.as_mut_ptr();
    let res = unsafe { libc::epoll_wait(epfd, array_ptr, maxsize.try_into().unwrap(), timeout) };
    if res < 0 {
        panic!("epoll_wait failed: {}", std::io::Error::last_os_error());
    }
    assert_eq!(
        res,
        expected_notifications.len().try_into().unwrap(),
        "got wrong number of notifications"
    );
    let slice = unsafe { std::slice::from_raw_parts(array_ptr, res.try_into().unwrap()) };
    for (return_event, expected_event) in slice.iter().zip(expected_notifications.iter()) {
        let event = return_event.events;
        let data = return_event.u64;
        assert_eq!(event, expected_event.0, "got wrong events");
        assert_eq!(data, expected_event.1, "got wrong data");
    }
}

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
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fd1 as u64 };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd1, &mut ev) };
    assert_eq!(res, 0);
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fd2 as u64 };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd2, &mut ev) };
    assert_eq!(res, 0);

    // Consume the initial events.
    let expected = [(libc::EPOLLOUT as u32, fd1 as u64), (libc::EPOLLOUT as u32, fd2 as u64)];
    check_epoll_wait::<8>(epfd, &expected, -1);

    let thread1 = thread::spawn(move || {
        check_epoll_wait::<2>(epfd, &expected, -1);
    });
    let thread2 = thread::spawn(move || {
        check_epoll_wait::<2>(epfd, &expected, -1);
        //~^ERROR: deadlocked
    });
    // Yield so the threads are both blocked.
    thread::yield_now();

    // Create two events at once.
    libc_utils::write_all_from_slice(fd1, &0_u64.to_ne_bytes()).unwrap();

    thread1.join().unwrap();
    thread2.join().unwrap();
}
