//@compile-flags: -Zmiri-deterministic-concurrency
//~^ERROR: deadlocked
//~^^ERROR: deadlocked
//@only-target: linux android illumos
//@error-in-other-file: deadlock

use std::convert::TryInto;
use std::thread::spawn;

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

    let expected_event = u32::try_from(libc::EPOLLIN | libc::EPOLLOUT).unwrap();
    let expected_value = fds[0] as u64;
    let thread1 = spawn(move || {
        check_epoll_wait::<1>(epfd, &[(expected_event, expected_value)], -1);
        //~^ERROR: deadlocked
    });
    let thread2 = spawn(move || {
        check_epoll_wait::<1>(epfd, &[(expected_event, expected_value)], -1);
    });

    let thread3 = spawn(move || {
        // Just a single write, so we only wake up one of them.
        let data = "abcde".as_bytes().as_ptr();
        let res = unsafe { libc::write(fds[1], data as *const libc::c_void, 5) };
        assert!(res > 0 && res <= 5);
    });

    thread1.join().unwrap();
    thread2.join().unwrap();
    thread3.join().unwrap();
}
