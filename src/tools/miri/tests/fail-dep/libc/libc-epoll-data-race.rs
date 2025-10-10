//! This ensures that when an epoll_wait wakes up and there are multiple events,
//! and we only read one of them, we do not synchronize with the other events
//! and therefore still report a data race for things that need to see the second event
//! to be considered synchronized.
//@only-target: linux android illumos
// ensure deterministic schedule
//@compile-flags: -Zmiri-deterministic-concurrency

use std::convert::TryInto;
use std::thread;
use std::thread::spawn;

#[path = "../../utils/libc.rs"]
mod libc_utils;

#[track_caller]
fn check_epoll_wait<const N: usize>(epfd: i32, expected_notifications: &[(u32, u64)]) {
    let epoll_event = libc::epoll_event { events: 0, u64: 0 };
    let mut array: [libc::epoll_event; N] = [epoll_event; N];
    let maxsize = N;
    let array_ptr = array.as_mut_ptr();
    let res = unsafe { libc::epoll_wait(epfd, array_ptr, maxsize.try_into().unwrap(), 0) };
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

fn main() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create two socketpair instances.
    let mut fds_a = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds_a.as_mut_ptr()) };
    assert_eq!(res, 0);

    let mut fds_b = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds_b.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Register both pipe read ends.
    let mut ev = libc::epoll_event {
        events: (libc::EPOLLIN | libc::EPOLLET) as _,
        u64: u64::try_from(fds_a[1]).unwrap(),
    };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds_a[1], &mut ev) };
    assert_eq!(res, 0);

    let mut ev = libc::epoll_event {
        events: (libc::EPOLLIN | libc::EPOLLET) as _,
        u64: u64::try_from(fds_b[1]).unwrap(),
    };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds_b[1], &mut ev) };
    assert_eq!(res, 0);

    static mut VAL_ONE: u8 = 40; // This one will be read soundly.
    static mut VAL_TWO: u8 = 50; // This one will be read unsoundly.
    let thread1 = spawn(move || {
        unsafe { VAL_ONE = 41 };

        let data = "abcde".as_bytes().as_ptr();
        let res = unsafe { libc_utils::write_all(fds_a[0], data as *const libc::c_void, 5) };
        assert_eq!(res, 5);

        unsafe { VAL_TWO = 51 };

        let res = unsafe { libc_utils::write_all(fds_b[0], data as *const libc::c_void, 5) };
        assert_eq!(res, 5);
    });
    thread::yield_now();

    // With room for one event: check result from epoll_wait.
    let expected_event = u32::try_from(libc::EPOLLIN).unwrap();
    let expected_value = u64::try_from(fds_a[1]).unwrap();
    check_epoll_wait::<1>(epfd, &[(expected_event, expected_value)]);

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
