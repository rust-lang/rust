//@only-target: linux android illumos
//@revisions: edge_triggered level_triggered
// We need to control yielding.
//@compile-flags: -Zmiri-deterministic-concurrency
//@run-native

use std::convert::TryInto;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

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
    waiting_threads_unblocked_after_epoll_close();
    waiting_threads_unblocked_after_socketpair_close();
    epoll_blocking_watching_fd_that_is_being_closed();
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

    // Wait for EPOLLIN (it will not happen).
    epoll_ctl_add(epfd, fd, EPOLLIN | EPOLLET_OR_ZERO).unwrap();
    let start = Instant::now();
    check_epoll_wait(epfd, &[], 10);
    assert!(start.elapsed() >= Duration::from_millis(10));

    // The second test behaves differently between level- and edge-triggered epoll.
    // We get a new epoll instance for this.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Register eventfd with epoll.
    epoll_ctl_add(epfd, fd, EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // epoll_wait to clear notification.
    check_epoll_wait(epfd, &[Ev { events: EPOLLOUT, data: fd }], -1);

    if cfg!(edge_triggered) {
        // This epoll wait blocks, and timeout without notification.
        let start = Instant::now();
        check_epoll_wait(epfd, &[], 10);
        assert!(start.elapsed() >= Duration::from_millis(10));
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
        thread::sleep(Duration::from_millis(10));
        // Since we slept, we should only get here after the other thread blocks.
        write_all(fds[1], b"abcde").unwrap();
    });

    if cfg!(edge_triggered) {
        // Edge-triggered epoll will block until the write succeeds and the buffer
        // becomes readable. This is because we already read the writable edge
        // before so at the time of calling `epoll_wait` there is no active readiness.
        check_epoll_wait(epfd, &[Ev { events: EPOLLIN | EPOLLOUT, data: fds[0] }], 100);
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

// This test shows that there is no data race when synchronizing through an epoll wakeup.
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
        eventfd::write_val(fd, 1).unwrap();
    });
    thread::sleep(Duration::from_millis(10));
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
    thread::sleep(Duration::from_millis(10));

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
    // (Can't use helper because we are gathering the events from both threads before comparing.)
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
    thread::sleep(Duration::from_millis(10));

    // Trigger the eventfd. This triggers two events at once!
    eventfd::write_val(fd1, 0).unwrap();

    // Both threads should have been woken up so that both events can be consumed.
    let e1 = t1.join().unwrap();
    let e2 = t2.join().unwrap();

    // In both modes we should get both events across the two threads.
    assert!(expected == [e1, e2] || expected == [e2, e1]);
}

/// Test that threads which are waiting on an epoll are unblocked when a registered interest
/// is fulfilled, even when the epoll file _descriptor_ they block on got closed in the
/// mean time.
fn waiting_threads_unblocked_after_epoll_close() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Add `fds[0]` to epoll with readable interest.
    epoll_ctl_add(epfd, fds[0], EPOLLIN | EPOLLET_OR_ZERO).unwrap();

    let t1 = thread::spawn(move || {
        // Sleep 10ms to make sure the other thread is blocked.
        thread::sleep(Duration::from_millis(10));

        // Close epoll file descriptor.
        unsafe { errno_check(libc::close(epfd)) };

        // Write some data into `fds[1]` which should make `fds[0]` readable.
        write_all(fds[1], b"abcde").unwrap();
    });

    // Indefinitely block until `fds[0]` becomes readable.
    check_epoll_wait(epfd, &[Ev { events: EPOLLIN, data: fds[0] }], -1);

    t1.join().unwrap();
}

/// Check correct behavior when a socketpair FD is closed while a thread blocks on it.
/// That thread keeps a reference so it is only really closed when that thread wakes up
/// again, and at that point an epoll notification should be triggered.
fn waiting_threads_unblocked_after_socketpair_close() {
    // There are multiple variants of this test.
    for variant in 0..=1 {
        // Create a socketpair instance.
        let mut fds = [-1, -1];
        errno_check(unsafe {
            libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr())
        });

        // Create an epoll instance, register `fds[1]`.
        let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();
        epoll_ctl_add(epfd, fds[1], EPOLLIN | EPOLLET_OR_ZERO).unwrap();

        // And an atomic variable for synchronization.
        let flag = AtomicBool::new(false);

        thread::scope(|s| {
            // Thread 1 will block reading on fds[0], thread 2 blocks on epoll for fds[1].
            s.spawn(|| {
                let data = read_exact_array::<4>(fds[0]).unwrap();
                assert_eq!(&data, b"1234");
                flag.store(true, Ordering::Relaxed);
            });
            if cfg!(level_triggered) || variant == 0 {
                // In variant 1, the event is consumed below (for edge-triggered).
                s.spawn(|| {
                    // Indefinitely block until `fds[1]` becomes readable.
                    check_epoll_wait(epfd, &[Ev { events: EPOLLIN | EPOLLHUP, data: fds[1] }], -1);
                    flag.store(true, Ordering::Relaxed);
                });
            }

            // Let the threads go and do their setup.
            thread::sleep(Duration::from_millis(10));

            // Once they did their setup, close fds[0] (will not really be closed since there is a
            // thread blocked on it).
            unsafe { errno_check(libc::close(fds[0])) };

            // This should *not* yet wake up anyone. So we wait a bit and check the flag.
            thread::sleep(Duration::from_millis(10));
            assert_eq!(flag.load(Ordering::Relaxed), false);

            // ... and then write to fds[1] to make it readable.
            write_all(fds[1], b"1234").unwrap();

            // We want to both test "delayed readiness processed by scheduler" and "delayed
            // readiness processed by epoll_wait", so we have two variants of this test.
            if variant == 1 {
                // Now readiness should be updated, even before we schedule to another thread. Needs to be
                // non-blocking to hit what used to be a buggy codepath! Interestingly, we do *not* always
                // immediately see the new events on native runs -- it's almost as if Linux also delays
                // updating the readiness. We still want to update readiness immediately in Miri as
                // otherwise we'd have to give up on a nice strong invariant and disable a sanity check.
                // So we give the native kernel a bit of time to update the readiness.
                if !cfg!(miri) {
                    // Give the kernel some time to process.
                    thread::sleep(Duration::from_millis(10));
                }
                check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLIN | EPOLLHUP, data: fds[1] }]);
            }
        });
    }
}

/// Ensure correct behavior when we block on epoll watching an FD that is being closed.
fn epoll_blocking_watching_fd_that_is_being_closed() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create an eventfd instance and register it with epoll.
    let fd = errno_result(unsafe { libc::eventfd(0, 0) }).unwrap();
    epoll_ctl_add(epfd, fd, EPOLLIN | EPOLLET_OR_ZERO).unwrap();

    thread::scope(|s| {
        s.spawn(|| {
            // Block on the epoll.
            let start = Instant::now();
            check_epoll_wait(epfd, &[], 10);
            assert!(start.elapsed() >= Duration::from_millis(10));
        });

        // Let the thread go and do its setup.
        thread::sleep(Duration::from_millis(10));

        // Now that the thread is blocked, close the eventfd.
        unsafe { errno_check(libc::close(fd)) };

        // The epoll_wait above should time out. Being closed does not generate any events.
    });
}
