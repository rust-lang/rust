//@only-target: linux android illumos
// test_epoll_block_then_unblock and test_epoll_race depend on a deterministic schedule.
//@compile-flags: -Zmiri-deterministic-concurrency
//@revisions: edge_triggered level_triggered
//@run-native

use std::convert::TryInto;
use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::epoll::*;
use libc_utils::*;

/// When the `edge_triggered` revision is active, this is EPOLLET, otherwise
/// it's zero which means we perform level-triggered epolls.
const EPOLLET_OR_ZERO: libc::c_int = if cfg!(edge_triggered) { EPOLLET } else { 0 };

// This is a set of testcases for blocking epoll.

fn main() {
    test_epoll_block_without_notification();
    test_epoll_block_then_unblock();
    test_notification_after_timeout();
    test_epoll_race();
    wakeup_on_new_interest();
    multiple_events_wake_multiple_threads();
}

// This test allows edge-triggered epoll_wait to block and then unblock
// without notification because the timeout expired.
// The level-triggered epoll_wait should not block.
fn test_epoll_block_without_notification() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create an eventfd instances.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();

    // Register eventfd with epoll.
    epoll_ctl_add(epfd, fd, EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // epoll_wait to clear notification.
    check_epoll_wait(epfd, &[Ev { events: EPOLLOUT, data: fd }], -1);

    if cfg!(edge_triggered) {
        // This epoll wait blocks, and timeout without notification.
        check_epoll_wait(epfd, &[], 5);
    } else {
        // In level-triggered mode we should receive the same events
        // as before without timing out.
        check_epoll_wait(epfd, &[Ev { events: EPOLLOUT, data: fd }], -1);
    }
}

// This test triggers notification and unblocks the edge-triggered epoll_wait
// before the timeout exceeds.
// The level-triggered epoll_wait should not block.
fn test_epoll_block_then_unblock() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Register one side of the socketpair with epoll.
    epoll_ctl_add(epfd, fds[0], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // epoll_wait to clear notification.
    check_epoll_wait(epfd, &[Ev { events: EPOLLOUT, data: fds[0] }], -1);

    let thread1 = thread::spawn(move || {
        thread::yield_now();
        // Due to deterministic concurrency, we'll only get here when the other thread blocks.
        write_all(fds[1], b"abcde").unwrap();
    });

    if cfg!(edge_triggered) {
        // Edge-triggered epoll will block until the write succeeds and the buffer
        // becomes readable. This is because we already read the writable edge
        // before so at the time of calling `epoll_wait` there is no active readiness.
        check_epoll_wait(epfd, &[Ev { events: EPOLLIN | EPOLLOUT, data: fds[0] }], 10);
    } else {
        // Level-triggered epoll won't wait for the write to succeed because
        // _some_ readiness is already set (in this case the EPOLLOUT).
        check_epoll_wait(epfd, &[Ev { events: EPOLLOUT, data: fds[0] }], -1);
    }

    thread1.join().unwrap();
}

// This test triggers a notification after epoll_wait times out in edge-triggered mode.
// In level-triggered the epoll_wait should not time out.
fn test_notification_after_timeout() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Register one side of the socketpair with epoll.
    epoll_ctl_add(epfd, fds[0], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // epoll_wait to clear notification.
    check_epoll_wait(epfd, &[Ev { events: EPOLLOUT, data: fds[0] }], -1);

    if cfg!(edge_triggered) {
        // Edge-triggered epoll wait times out without notification because
        // we just processed the edge.
        check_epoll_wait(epfd, &[], 10);
    } else {
        // Level-triggered epoll just returns the same events as before
        // without blocking.
        check_epoll_wait(epfd, &[Ev { events: EPOLLOUT, data: fds[0] }], -1);
    }

    // Trigger epoll notification after timeout.
    write_all(fds[1], b"abcde").unwrap();

    // Check the result of the notification.
    check_epoll_wait(epfd, &[Ev { events: EPOLLIN | EPOLLOUT, data: fds[0] }], 10);
}

// This test shows a data race before epoll had vector clocks added.
fn test_epoll_race() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create an eventfd instance.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();

    // Register eventfd with the epoll instance.
    epoll_ctl_add(epfd, fd, EPOLLIN | EPOLLET_OR_ZERO).unwrap();

    static mut VAL: u8 = 0;
    let thread1 = thread::spawn(move || {
        // Write to the static mut variable.
        unsafe { VAL = 1 };
        // Write to the eventfd instance.
        write_all(fd, &1_u64.to_ne_bytes()).unwrap();
    });
    thread::yield_now();
    // epoll_wait for EPOLLIN.
    check_epoll_wait(epfd, &[Ev { events: EPOLLIN, data: fd }], -1);
    // Read from the static mut variable.
    assert_eq!(unsafe { VAL }, 1);
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
    write_all(fds[0], b"abcde").unwrap();

    // Block a thread on the epoll instance.
    let t = std::thread::spawn(move || {
        check_epoll_wait(epfd, &[Ev { events: EPOLLIN | EPOLLOUT, data: fds[1] }], -1);
    });
    // Ensure the thread is blocked.
    std::thread::yield_now();

    // Register fd[1] with EPOLLIN|EPOLLOUT|EPOLLRDHUP (and EPOLLET if we're in the
    // `edge_triggered` revision).
    epoll_ctl_add(epfd, fds[1], EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET_OR_ZERO).unwrap();

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
    epoll_ctl_add(epfd, fd1, EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();
    epoll_ctl_add(epfd, fd2, EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Consume the initial events.
    let expected = [Ev { events: EPOLLOUT, data: fd1 }, Ev { events: EPOLLOUT, data: fd2 }];
    check_epoll_wait(epfd, &expected, -1);

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
    write_all(fd1, &0_u64.to_ne_bytes()).unwrap();

    // Both threads should have been woken up so that both events can be consumed.
    let e1 = t1.join().unwrap();
    let e2 = t2.join().unwrap();

    // In both modes we should get both events across the two threads.
    assert!(expected == [e1, e2] || expected == [e2, e1]);
}
