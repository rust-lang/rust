//@only-target: linux android illumos
// test_epoll_block_then_unblock and test_epoll_race depend on a deterministic schedule.
//@compile-flags: -Zmiri-deterministic-concurrency

use std::convert::TryInto;
use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::epoll::*;
use libc_utils::*;

// This is a set of testcases for blocking epoll.

fn main() {
    test_epoll_block_without_notification();
    test_epoll_block_then_unblock();
    test_notification_after_timeout();
    test_epoll_race();
    wakeup_on_new_interest();
    multiple_events_wake_multiple_threads();
}

// This test allows epoll_wait to block, then unblock without notification.
fn test_epoll_block_without_notification() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create an eventfd instances.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();

    // Register eventfd with epoll.
    epoll_ctl_add(epfd, fd, libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET).unwrap();

    // epoll_wait to clear notification.
    check_epoll_wait::<1>(epfd, &[Ev { events: libc::EPOLLOUT, data: fd }], 0);

    // This epoll wait blocks, and timeout without notification.
    check_epoll_wait::<1>(epfd, &[], 5);
}

// This test triggers notification and unblocks the epoll_wait before timeout.
fn test_epoll_block_then_unblock() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Register one side of the socketpair with epoll.
    epoll_ctl_add(epfd, fds[0], libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET).unwrap();

    // epoll_wait to clear notification.
    check_epoll_wait::<1>(epfd, &[Ev { events: libc::EPOLLOUT, data: fds[0] }], 0);

    // epoll_wait before triggering notification so it will block then get unblocked before timeout.
    let thread1 = thread::spawn(move || {
        thread::yield_now();
        write_all_from_slice(fds[1], b"abcde").unwrap();
    });
    check_epoll_wait::<1>(epfd, &[Ev { events: libc::EPOLLIN | libc::EPOLLOUT, data: fds[0] }], 10);
    thread1.join().unwrap();
}

// This test triggers a notification after epoll_wait times out.
fn test_notification_after_timeout() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Register one side of the socketpair with epoll.
    epoll_ctl_add(epfd, fds[0], libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET).unwrap();

    // epoll_wait to clear notification.
    check_epoll_wait::<1>(epfd, &[Ev { events: libc::EPOLLOUT, data: fds[0] }], 0);

    // epoll_wait timeouts without notification.
    check_epoll_wait::<1>(epfd, &[], 10);

    // Trigger epoll notification after timeout.
    write_all_from_slice(fds[1], b"abcde").unwrap();

    // Check the result of the notification.
    check_epoll_wait::<1>(epfd, &[Ev { events: libc::EPOLLIN | libc::EPOLLOUT, data: fds[0] }], 10);
}

// This test shows a data_race before epoll had vector clocks added.
fn test_epoll_race() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create an eventfd instance.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();

    // Register eventfd with the epoll instance.
    epoll_ctl_add(epfd, fd, libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET).unwrap();

    static mut VAL: u8 = 0;
    let thread1 = thread::spawn(move || {
        // Write to the static mut variable.
        unsafe { VAL = 1 };
        // Write to the eventfd instance.
        write_all_from_slice(fd, &1_u64.to_ne_bytes()).unwrap();
    });
    thread::yield_now();
    // epoll_wait for the event to happen.
    check_epoll_wait::<8>(epfd, &[Ev { events: (libc::EPOLLIN | libc::EPOLLOUT), data: fd }], -1);
    // Read from the static mut variable.
    #[allow(static_mut_refs)]
    unsafe {
        assert_eq!(VAL, 1)
    };
    thread1.join().unwrap();
}

/// Ensure that a blocked thread gets woken up when new interested are registered with the
/// epoll it is blocked on.
fn wakeup_on_new_interest() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Write to fd[0]
    write_all_from_slice(fds[0], b"abcde").unwrap();

    // Block a thread on the epoll instance.
    let t = std::thread::spawn(move || {
        check_epoll_wait::<8>(
            epfd,
            &[Ev { events: libc::EPOLLIN | libc::EPOLLOUT, data: fds[1] }],
            -1,
        );
    });
    // Ensure the thread is blocked.
    std::thread::yield_now();

    // Register fd[1] with EPOLLIN|EPOLLOUT|EPOLLET|EPOLLRDHUP
    epoll_ctl_add(epfd, fds[1], libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET | libc::EPOLLRDHUP)
        .unwrap();

    // This should wake up the thread.
    t.join().unwrap();
}

/// Ensure that if a single operation triggers multiple events, we wake up enough threads
/// to consume them all.
fn multiple_events_wake_multiple_threads() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create an eventfd instance.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd1 = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();
    // Make a duplicate so that we have two file descriptors for the same file description.
    let fd2 = errno_result(unsafe { libc::dup(fd1) }).unwrap();

    // Register both with epoll.
    epoll_ctl_add(epfd, fd1, libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET).unwrap();
    epoll_ctl_add(epfd, fd2, libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET).unwrap();

    // Consume the initial events.
    let expected =
        [Ev { events: libc::EPOLLOUT, data: fd1 }, Ev { events: libc::EPOLLOUT, data: fd2 }];
    check_epoll_wait::<8>(epfd, &expected, -1);

    // Block two threads on the epoll, both wanting to get just one event.
    let t1 = thread::spawn(move || {
        let mut e = libc::epoll_event { events: 0, u64: 0 };
        let res = unsafe { libc::epoll_wait(epfd, &raw mut e, 1, -1) };
        assert!(res == 1);
        Ev { events: e.events.cast_signed(), data: e.u64.try_into().unwrap() }
    });
    let t2 = thread::spawn(move || {
        let mut e = libc::epoll_event { events: 0, u64: 0 };
        let res = unsafe { libc::epoll_wait(epfd, &raw mut e, 1, -1) };
        assert!(res == 1);
        Ev { events: e.events.cast_signed(), data: e.u64.try_into().unwrap() }
    });
    // Yield so both threads are waiting now.
    thread::yield_now();

    // Trigger the eventfd. This triggers two events at once!
    write_all_from_slice(fd1, &0_u64.to_ne_bytes()).unwrap();

    // Both threads should have been woken up so that both events can be consumed.
    let e1 = t1.join().unwrap();
    let e2 = t2.join().unwrap();
    // Ensure that across the two threads we got both events.
    assert!(expected == [e1, e2] || expected == [e2, e1]);
}
