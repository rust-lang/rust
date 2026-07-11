//@ignore-target: windows # no libc
//@compile-flags: -Zmiri-deterministic-concurrency
//@run-native

use std::thread;
use std::time::{Duration, Instant};

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

const TEST_BYTES: &[u8] = b"these are some test bytes!";

fn main() {
    test_poll_returns_masked();
    test_poll_already_fulfilled();
    test_poll_block_without_events();
    test_poll_readiness_update();
    test_poll_negative_fd_interest();
    test_poll_invalid_non_negative_fd_interest();
}

/// Test that the readiness written into the `revents` field on an interest
/// is masked with the relevant readiness of the interest.
fn test_poll_returns_masked() {
    let mut fds = [-1, -1];
    unsafe { errno_check(libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr())) };

    // Write some data into `fds[1]` such that `fds[0]` becomes readable.
    unsafe {
        errno_result(libc::write(fds[1], TEST_BYTES.as_ptr().cast(), TEST_BYTES.len())).unwrap()
    };

    let mut interests = [libc::pollfd { fd: fds[0], events: libc::POLLOUT, revents: 0 }];
    let ready = unsafe {
        errno_result(libc::poll(interests.as_mut_ptr(), interests.len() as libc::nfds_t, -1))
            .unwrap()
    };
    assert_eq!(ready, 1);
    // Ensure that the correct `revents` has been set.
    // Because we're only interested in the writable readiness, the readable
    // readiness should not be written into the `revents` field.
    assert_eq!(interests[0].revents, libc::POLLOUT);
}

/// Test that the `poll` call returns a ready event when the
/// provided interest is already fulfilled before calling `poll`.
fn test_poll_already_fulfilled() {
    let mut fds = [-1, -1];
    unsafe { errno_check(libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr())) };

    let mut interests =
        [libc::pollfd { fd: fds[0], events: libc::POLLIN | libc::POLLOUT, revents: 0 }];
    let ready = unsafe {
        errno_result(libc::poll(interests.as_mut_ptr(), interests.len() as libc::nfds_t, -1))
            .unwrap()
    };
    assert_eq!(ready, 1);
    // Ensure that the correct `revents` has been set.
    assert_eq!(interests[0].revents, libc::POLLOUT);
}

/// Test that the `poll` blocks and returns zero when
/// none of the provided interests get fulfilled.
fn test_poll_block_without_events() {
    let mut fds = [-1, -1];
    unsafe { errno_check(libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr())) };

    let mut interests = [libc::pollfd { fd: fds[0], events: libc::POLLIN, revents: 1 }];
    let before = Instant::now();
    let ready = unsafe {
        errno_result(libc::poll(interests.as_mut_ptr(), interests.len() as libc::nfds_t, 50))
            .unwrap()
    };
    assert_eq!(ready, 0);
    // Because the interest wasn't fulfilled, it's ready events should be zeroed.
    assert_eq!(interests[0].revents, 0);
    // Ensure that the `poll` blocked at least for 50ms.
    assert!(before.elapsed() > Duration::from_millis(50))
}

/// Test that the `poll` blocks when the requested interests are not
/// fulfilled at creation. This also tests that the `poll` unblocks
/// once the readiness of a registered fd changes.
fn test_poll_readiness_update() {
    let mut fds = [-1, -1];
    unsafe { errno_check(libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr())) };

    let t1 = thread::spawn(move || {
        // Yield to main thread to ensure it's blocked on `poll` before
        // we write into the socket.
        thread::sleep(Duration::from_millis(10));

        unsafe {
            errno_result(libc::write(fds[1], TEST_BYTES.as_ptr().cast(), TEST_BYTES.len())).unwrap()
        }
    });

    let mut interests = [libc::pollfd { fd: fds[0], events: libc::POLLIN, revents: 0 }];
    let ready = unsafe {
        errno_result(libc::poll(interests.as_mut_ptr(), interests.len() as libc::nfds_t, 150))
            .unwrap()
    };
    assert_eq!(ready, 1);
    // Ensure that the correct `revents` has been set.
    assert_eq!(interests[0].revents, libc::POLLIN);

    t1.join().unwrap();
}

/// Test that interests in negative file descriptors are ignored and
/// aren't treated as fulfilled.
fn test_poll_negative_fd_interest() {
    let mut interests = [libc::pollfd { fd: -1, events: libc::POLLHUP, revents: 0 }];

    let ready = unsafe {
        errno_result(libc::poll(interests.as_mut_ptr(), interests.len() as libc::nfds_t, 10))
            .unwrap()
    };
    // Interests in negative file descriptors should just be ignored but not be
    // treated as fulfilled. The `poll` should thus time out without any interest
    // being fulfilled.
    assert_eq!(ready, 0);
}

/// Test that the `poll` invocation correctly sets `revents` to POLLNVAL
/// when an invalid non-negative file descriptor is provided, and that
/// `poll` instantly returns.
fn test_poll_invalid_non_negative_fd_interest() {
    let mut interests = [libc::pollfd { fd: libc::c_int::MAX, events: libc::POLLHUP, revents: 0 }];

    // We provide an "infinite" timeout because interests in invalid non-negative file descriptors
    // are considered fulfilled and the `poll` invocation should thus instantly return.
    let ready = unsafe {
        errno_result(libc::poll(interests.as_mut_ptr(), interests.len() as libc::nfds_t, -1))
            .unwrap()
    };
    assert_eq!(ready, 1);
    // Ensure that the `revents` has correctly been set to POLLNVAL since this file
    // descriptor should not exist.
    assert_eq!(interests[0].revents, libc::POLLNVAL);
}
