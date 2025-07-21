//@only-target: linux android illumos
// test_epoll_block_then_unblock and test_epoll_race depend on a deterministic schedule.
//@compile-flags: -Zmiri-deterministic-concurrency

use std::convert::TryInto;
use std::thread;
use std::thread::spawn;

#[path = "../../utils/libc.rs"]
mod libc_utils;

// This is a set of testcases for blocking epoll.

fn main() {
    test_epoll_block_without_notification();
    test_epoll_block_then_unblock();
    test_notification_after_timeout();
    test_epoll_race();
}

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

// This test allows epoll_wait to block, then unblock without notification.
fn test_epoll_block_without_notification() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create an eventfd instances.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };

    // Register eventfd with epoll.
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fd as u64 };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
    assert_eq!(res, 0);

    // epoll_wait to clear notification.
    let expected_event = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value = fd as u64;
    check_epoll_wait::<1>(epfd, &[(expected_event, expected_value)], 0);

    // This epoll wait blocks, and timeout without notification.
    check_epoll_wait::<1>(epfd, &[], 5);
}

// This test triggers notification and unblocks the epoll_wait before timeout.
fn test_epoll_block_then_unblock() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Register one side of the socketpair with epoll.
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fds[0] as u64 };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[0], &mut ev) };
    assert_eq!(res, 0);

    // epoll_wait to clear notification.
    let expected_event = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value = fds[0] as u64;
    check_epoll_wait::<1>(epfd, &[(expected_event, expected_value)], 0);

    // epoll_wait before triggering notification so it will block then get unblocked before timeout.
    let expected_event = u32::try_from(libc::EPOLLIN | libc::EPOLLOUT).unwrap();
    let expected_value = fds[0] as u64;
    let thread1 = spawn(move || {
        thread::yield_now();
        let data = "abcde".as_bytes().as_ptr();
        let res = unsafe { libc_utils::write_all(fds[1], data as *const libc::c_void, 5) };
        assert_eq!(res, 5);
    });
    check_epoll_wait::<1>(epfd, &[(expected_event, expected_value)], 10);
    thread1.join().unwrap();
}

// This test triggers a notification after epoll_wait times out.
fn test_notification_after_timeout() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Register one side of the socketpair with epoll.
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fds[0] as u64 };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[0], &mut ev) };
    assert_eq!(res, 0);

    // epoll_wait to clear notification.
    let expected_event = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value = fds[0] as u64;
    check_epoll_wait::<1>(epfd, &[(expected_event, expected_value)], 0);

    // epoll_wait timeouts without notification.
    check_epoll_wait::<1>(epfd, &[], 10);

    // Trigger epoll notification after timeout.
    let data = "abcde".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[1], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);

    // Check the result of the notification.
    let expected_event = u32::try_from(libc::EPOLLIN | libc::EPOLLOUT).unwrap();
    let expected_value = fds[0] as u64;
    check_epoll_wait::<1>(epfd, &[(expected_event, expected_value)], 10);
}

// This test shows a data_race before epoll had vector clocks added.
fn test_epoll_race() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create an eventfd instance.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };

    // Register eventfd with the epoll instance.
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fd as u64 };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
    assert_eq!(res, 0);

    static mut VAL: u8 = 0;
    let thread1 = thread::spawn(move || {
        // Write to the static mut variable.
        unsafe { VAL = 1 };
        // Write to the eventfd instance.
        let sized_8_data: [u8; 8] = 1_u64.to_ne_bytes();
        let res = unsafe { libc::write(fd, sized_8_data.as_ptr() as *const libc::c_void, 8) };
        // write returns number of bytes written, which is always 8.
        assert_eq!(res, 8);
    });
    thread::yield_now();
    // epoll_wait for the event to happen.
    let expected_event = u32::try_from(libc::EPOLLIN | libc::EPOLLOUT).unwrap();
    let expected_value = u64::try_from(fd).unwrap();
    check_epoll_wait::<8>(epfd, &[(expected_event, expected_value)], -1);
    // Read from the static mut variable.
    #[allow(static_mut_refs)]
    unsafe {
        assert_eq!(VAL, 1)
    };
    thread1.join().unwrap();
}
