//@only-target: linux android illumos

use std::convert::TryInto;

#[path = "../../utils/libc.rs"]
mod libc_utils;

fn main() {
    test_epoll_socketpair();
    test_epoll_socketpair_both_sides();
    test_socketpair_read();
    test_epoll_eventfd();

    test_event_overwrite();
    test_not_fully_closed_fd();
    test_closed_fd();
    test_two_epoll_instance();
    test_no_notification_for_unregister_flag();
    test_epoll_ctl_mod();
    test_epoll_ctl_del();
    test_two_same_fd_in_same_epoll_instance();
    test_epoll_wait_maxevent_zero();
    test_socketpair_epollerr();
    test_epoll_lost_events();
    test_ready_list_fetching_logic();
    test_epoll_ctl_epfd_equal_fd();
    test_epoll_ctl_notification();
    test_issue_3858();
}

// Using `as` cast since `EPOLLET` wraps around
const EPOLL_IN_OUT_ET: u32 = (libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET) as _;

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

fn test_epoll_socketpair() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Write to fd[0]
    let data = "abcde".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[0], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);

    // Register fd[1] with EPOLLIN|EPOLLOUT|EPOLLET|EPOLLRDHUP
    let mut ev = libc::epoll_event {
        events: (libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET | libc::EPOLLRDHUP) as _,
        u64: u64::try_from(fds[1]).unwrap(),
    };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[1], &mut ev) };
    assert_eq!(res, 0);

    // Check result from epoll_wait.
    let expected_event = u32::try_from(libc::EPOLLIN | libc::EPOLLOUT).unwrap();
    let expected_value = u64::try_from(fds[1]).unwrap();
    check_epoll_wait::<8>(epfd, &[(expected_event, expected_value)]);

    // Check that this is indeed using "ET" (edge-trigger) semantics: a second epoll should return nothing.
    check_epoll_wait::<8>(epfd, &[]);

    // Write some more to fd[0].
    let data = "abcde".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[0], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);

    // This did not change the readiness of fd[1]. And yet, we're seeing the event reported
    // again by the kernel, so Miri does the same.
    check_epoll_wait::<8>(epfd, &[(expected_event, expected_value)]);

    // Close the peer socketpair.
    let res = unsafe { libc::close(fds[0]) };
    assert_eq!(res, 0);

    // Check result from epoll_wait.
    // We expect to get a read, write, HUP notification from the close since closing an FD always unblocks reads and writes on its peer.
    let expected_event =
        u32::try_from(libc::EPOLLRDHUP | libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLHUP).unwrap();
    let expected_value = u64::try_from(fds[1]).unwrap();
    check_epoll_wait::<8>(epfd, &[(expected_event, expected_value)]);
}

// This test first registers a file description with a flag that does not lead to notification,
// then EPOLL_CTL_MOD to add another flag that will lead to notification.
fn test_epoll_ctl_mod() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Register fd[1] with EPOLLIN|EPOLLET.
    let mut ev = libc::epoll_event {
        events: (libc::EPOLLIN | libc::EPOLLET) as _,
        u64: u64::try_from(fds[1]).unwrap(),
    };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[1], &mut ev) };
    assert_eq!(res, 0);

    // Check result from epoll_wait. No notification would be returned.
    check_epoll_wait::<8>(epfd, &[]);

    // Use EPOLL_CTL_MOD to change to EPOLLOUT flag.
    let mut ev = libc::epoll_event {
        events: (libc::EPOLLOUT | libc::EPOLLET) as _,
        u64: u64::try_from(fds[1]).unwrap(),
    };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_MOD, fds[1], &mut ev) };
    assert_eq!(res, 0);

    // Check result from epoll_wait. EPOLLOUT notification is expected.
    let expected_event = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value = u64::try_from(fds[1]).unwrap();
    check_epoll_wait::<8>(epfd, &[(expected_event, expected_value)]);
}

fn test_epoll_ctl_del() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Write to fd[0]
    let data = "abcde".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[0], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);

    // Register fd[1] with EPOLLIN|EPOLLOUT|EPOLLET
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: u64::try_from(fds[1]).unwrap() };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[1], &mut ev) };
    assert_eq!(res, 0);

    // Test EPOLL_CTL_DEL.
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_DEL, fds[1], &mut ev) };
    assert_eq!(res, 0);
    check_epoll_wait::<8>(epfd, &[]);
}

// This test is for one fd registered under two different epoll instance.
fn test_two_epoll_instance() {
    // Create two epoll instance.
    let epfd1 = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd1, -1);
    let epfd2 = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd2, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Write to the socketpair.
    let data = "abcde".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[0], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);

    // Register one side of the socketpair with EPOLLIN | EPOLLOUT | EPOLLET.
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: u64::try_from(fds[1]).unwrap() };
    let res = unsafe { libc::epoll_ctl(epfd1, libc::EPOLL_CTL_ADD, fds[1], &mut ev) };
    assert_eq!(res, 0);
    let res = unsafe { libc::epoll_ctl(epfd2, libc::EPOLL_CTL_ADD, fds[1], &mut ev) };
    assert_eq!(res, 0);

    // Notification should be received from both instance of epoll.
    let expected_event = u32::try_from(libc::EPOLLIN | libc::EPOLLOUT).unwrap();
    let expected_value = u64::try_from(fds[1]).unwrap();
    check_epoll_wait::<8>(epfd1, &[(expected_event, expected_value)]);
    check_epoll_wait::<8>(epfd2, &[(expected_event, expected_value)]);
}

// This test is for two same file description registered under the same epoll instance through dup.
// Notification should be provided for both.
fn test_two_same_fd_in_same_epoll_instance() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Dup the fd.
    let newfd = unsafe { libc::dup(fds[1]) };
    assert_ne!(newfd, -1);

    // Register both fd to the same epoll instance.
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: 5 as u64 };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[1], &mut ev) };
    assert_eq!(res, 0);
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, newfd, &mut ev) };
    assert_eq!(res, 0);

    // Write to the socketpair.
    let data = "abcde".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[0], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);

    //Two notification should be received.
    let expected_event = u32::try_from(libc::EPOLLIN | libc::EPOLLOUT).unwrap();
    let expected_value = 5 as u64;
    check_epoll_wait::<8>(
        epfd,
        &[(expected_event, expected_value), (expected_event, expected_value)],
    );
}

fn test_epoll_eventfd() {
    // Create an eventfd instance.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };

    // Write to the eventfd instance.
    let sized_8_data: [u8; 8] = 1_u64.to_ne_bytes();
    let res = unsafe { libc_utils::write_all(fd, sized_8_data.as_ptr() as *const libc::c_void, 8) };
    assert_eq!(res, 8);

    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Register eventfd with EPOLLIN | EPOLLOUT | EPOLLET
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: u64::try_from(fd).unwrap() };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
    assert_eq!(res, 0);

    // Check result from epoll_wait.
    let expected_event = u32::try_from(libc::EPOLLIN | libc::EPOLLOUT).unwrap();
    let expected_value = u64::try_from(fd).unwrap();
    check_epoll_wait::<8>(epfd, &[(expected_event, expected_value)]);
}

// When read/write happened on one side of the socketpair, only the other side will be notified.
fn test_epoll_socketpair_both_sides() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Register both fd to the same epoll instance.
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fds[0] as u64 };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[0], &mut ev) };
    assert_eq!(res, 0);
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fds[1] as u64 };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[1], &mut ev) };
    assert_eq!(res, 0);

    // Write to fds[1].
    let data = "abcde".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[1], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);

    //Two notification should be received.
    let expected_event0 = u32::try_from(libc::EPOLLIN | libc::EPOLLOUT).unwrap();
    let expected_value0 = fds[0] as u64;
    let expected_event1 = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value1 = fds[1] as u64;
    check_epoll_wait::<8>(
        epfd,
        &[(expected_event0, expected_value0), (expected_event1, expected_value1)],
    );

    // Read from fds[0].
    let mut buf: [u8; 5] = [0; 5];
    let res =
        unsafe { libc_utils::read_all(fds[0], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
    assert_eq!(res, 5);
    assert_eq!(buf, "abcde".as_bytes());

    // Notification should be provided for fds[1].
    let expected_event = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value = fds[1] as u64;
    check_epoll_wait::<8>(epfd, &[(expected_event, expected_value)]);
}

// When file description is fully closed, epoll_wait should not provide any notification for
// that file description.
fn test_closed_fd() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create an eventfd instance.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };

    // Register eventfd with EPOLLIN | EPOLLOUT | EPOLLET
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: u64::try_from(fd).unwrap() };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
    assert_eq!(res, 0);

    // Write to the eventfd instance.
    let sized_8_data: [u8; 8] = 1_u64.to_ne_bytes();
    let res = unsafe { libc_utils::write_all(fd, sized_8_data.as_ptr() as *const libc::c_void, 8) };
    assert_eq!(res, 8);

    // Close the eventfd.
    let res = unsafe { libc::close(fd) };
    assert_eq!(res, 0);

    // No notification should be provided because the file description is closed.
    check_epoll_wait::<8>(epfd, &[]);
}

// When a certain file descriptor registered with epoll is closed, but the underlying file description
// is not closed, notification should still be provided.
//
// This is a quirk of epoll being described in https://man7.org/linux/man-pages/man7/epoll.7.html
// A file descriptor is removed from an interest list only after all the file descriptors
// referring to the underlying open file description have been closed.
fn test_not_fully_closed_fd() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create an eventfd instance.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };

    // Dup the fd.
    let newfd = unsafe { libc::dup(fd) };
    assert_ne!(newfd, -1);

    // Register eventfd with EPOLLIN | EPOLLOUT | EPOLLET
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: u64::try_from(fd).unwrap() };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
    assert_eq!(res, 0);

    // Close the original fd that being used to register with epoll.
    let res = unsafe { libc::close(fd) };
    assert_eq!(res, 0);

    // Notification should still be provided because the file description is not closed.
    let expected_event = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value = fd as u64;
    check_epoll_wait::<1>(epfd, &[(expected_event, expected_value)]);

    // Write to the eventfd instance to produce notification.
    let sized_8_data: [u8; 8] = 1_u64.to_ne_bytes();
    let res =
        unsafe { libc_utils::write_all(newfd, sized_8_data.as_ptr() as *const libc::c_void, 8) };
    assert_eq!(res, 8);

    // Close the dupped fd.
    let res = unsafe { libc::close(newfd) };
    assert_eq!(res, 0);

    // No notification should be provided.
    check_epoll_wait::<1>(epfd, &[]);
}

// Each time a notification is provided, it should reflect the file description's readiness
// at the moment the latest event occurred.
fn test_event_overwrite() {
    // Create an eventfd instance.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };

    // Write to the eventfd instance.
    let sized_8_data: [u8; 8] = 1_u64.to_ne_bytes();
    let res = unsafe { libc_utils::write_all(fd, sized_8_data.as_ptr() as *const libc::c_void, 8) };
    assert_eq!(res, 8);

    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Register eventfd with EPOLLIN | EPOLLOUT | EPOLLET
    let mut ev = libc::epoll_event {
        events: (libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET) as _,
        u64: u64::try_from(fd).unwrap(),
    };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
    assert_eq!(res, 0);

    // Read from the eventfd instance.
    let mut buf: [u8; 8] = [0; 8];
    let res = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), 8) };
    assert_eq!(res, 8);

    // Check result from epoll_wait.
    let expected_event = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value = u64::try_from(fd).unwrap();
    check_epoll_wait::<8>(epfd, &[(expected_event, expected_value)]);
}

// An epoll notification will be provided for every succesful read in a socketpair.
// This behaviour differs from the real system.
fn test_socketpair_read() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Register both fd to the same epoll instance.
    let mut ev = libc::epoll_event {
        events: (libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET) as _,
        u64: fds[0] as u64,
    };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[0], &mut ev) };
    assert_eq!(res, 0);
    let mut ev = libc::epoll_event {
        events: (libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET) as _,
        u64: fds[1] as u64,
    };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[1], &mut ev) };
    assert_eq!(res, 0);

    // Write 5 bytes to fds[1].
    let data = "abcde".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[1], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);

    //Two notification should be received.
    let expected_event0 = u32::try_from(libc::EPOLLIN | libc::EPOLLOUT).unwrap();
    let expected_value0 = fds[0] as u64;
    let expected_event1 = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value1 = fds[1] as u64;
    check_epoll_wait::<8>(
        epfd,
        &[(expected_event0, expected_value0), (expected_event1, expected_value1)],
    );

    // Read 3 bytes from fds[0].
    let mut buf: [u8; 3] = [0; 3];
    let res =
        unsafe { libc_utils::read_all(fds[0], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
    assert_eq!(res, 3);
    assert_eq!(buf, "abc".as_bytes());

    // Notification will be provided in Miri.
    // But in real systems, no notification will be provided here, since Linux prefers to avoid
    // wakeups that are likely to lead to only small amounts of data being read/written.
    // We make the test work in both cases, thus documenting the difference in behavior.
    let expected_event = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value = fds[1] as u64;
    if cfg!(miri) {
        check_epoll_wait::<8>(epfd, &[(expected_event, expected_value)]);
    } else {
        check_epoll_wait::<8>(epfd, &[]);
    }

    // Read until the buffer is empty.
    let mut buf: [u8; 2] = [0; 2];
    let res =
        unsafe { libc_utils::read_all(fds[0], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
    assert_eq!(res, 2);
    assert_eq!(buf, "de".as_bytes());

    // Notification will be provided.
    // In real system, notification will be provided too.
    let expected_event = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value = fds[1] as u64;
    check_epoll_wait::<8>(epfd, &[(expected_event, expected_value)]);
}

// This is to test whether flag that we don't register won't trigger notification.
fn test_no_notification_for_unregister_flag() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Register fd[0] with EPOLLOUT|EPOLLET.
    let mut ev = libc::epoll_event {
        events: (libc::EPOLLOUT | libc::EPOLLET) as _,
        u64: u64::try_from(fds[0]).unwrap(),
    };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[1], &mut ev) };
    assert_eq!(res, 0);

    // Write to fd[1].
    let data = "abcde".as_bytes().as_ptr();
    let res: i32 = unsafe {
        libc_utils::write_all(fds[1], data as *const libc::c_void, 5).try_into().unwrap()
    };
    assert_eq!(res, 5);

    // Check result from epoll_wait. Since we didn't register EPOLLIN flag, the notification won't
    // contain EPOLLIN even though fds[0] is now readable.
    let expected_event = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value = u64::try_from(fds[0]).unwrap();
    check_epoll_wait::<8>(epfd, &[(expected_event, expected_value)]);
}

fn test_epoll_wait_maxevent_zero() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);
    // It is ok to use a dangling pointer here because it will error out before the
    // pointer actually gets accessed.
    let array_ptr = std::ptr::without_provenance_mut::<libc::epoll_event>(0x100);
    let res = unsafe { libc::epoll_wait(epfd, array_ptr, 0, 0) };
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
    assert_eq!(res, -1);
}

fn test_socketpair_epollerr() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Write to fd[0]
    let data = "abcde".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[0], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);

    // Close fds[1].
    // EPOLLERR will be triggered if we close peer fd that still has data in its read buffer.
    let res = unsafe { libc::close(fds[1]) };
    assert_eq!(res, 0);

    // Register fd[1] with EPOLLIN|EPOLLOUT|EPOLLET|EPOLLRDHUP
    let mut ev = libc::epoll_event {
        events: (libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLET | libc::EPOLLRDHUP) as _,
        u64: u64::try_from(fds[1]).unwrap(),
    };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[0], &mut ev) };
    assert_ne!(res, -1);

    // Check result from epoll_wait.
    let expected_event = u32::try_from(
        libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLHUP | libc::EPOLLRDHUP | libc::EPOLLERR,
    )
    .unwrap();
    let expected_value = u64::try_from(fds[1]).unwrap();
    check_epoll_wait::<8>(epfd, &[(expected_event, expected_value)]);
}

// This is a test for https://github.com/rust-lang/miri/issues/3812,
// epoll can lose events if they don't fit in the output buffer.
fn test_epoll_lost_events() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Register both fd to the same epoll instance.
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fds[0] as u64 };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[0], &mut ev) };
    assert_eq!(res, 0);
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fds[1] as u64 };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[1], &mut ev) };
    assert_eq!(res, 0);

    //Two notification should be received. But we only provide buffer for one event.
    let expected_event0 = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value0 = fds[0] as u64;
    check_epoll_wait::<1>(epfd, &[(expected_event0, expected_value0)]);

    // Previous event should be returned for the second epoll_wait.
    let expected_event1 = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value1 = fds[1] as u64;
    check_epoll_wait::<1>(epfd, &[(expected_event1, expected_value1)]);
}

// This is testing if closing an fd that is already in ready list will cause an empty entry in
// returned notification.
// Related discussion in https://github.com/rust-lang/miri/pull/3818#discussion_r1720679440.
fn test_ready_list_fetching_logic() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Create two eventfd instances.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd0 = unsafe { libc::eventfd(0, flags) };
    let fd1 = unsafe { libc::eventfd(0, flags) };

    // Register both fd to the same epoll instance. At this point, both of them are on the ready list.
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fd0 as u64 };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd0, &mut ev) };
    assert_eq!(res, 0);
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fd1 as u64 };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd1, &mut ev) };
    assert_eq!(res, 0);

    // Close fd0 so the first entry in the ready list will be empty.
    let res = unsafe { libc::close(fd0) };
    assert_eq!(res, 0);

    // Notification for fd1 should be returned.
    let expected_event1 = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value1 = fd1 as u64;
    check_epoll_wait::<1>(epfd, &[(expected_event1, expected_value1)]);
}

// In epoll_ctl, if the value of epfd equals to fd, EINVAL should be returned.
fn test_epoll_ctl_epfd_equal_fd() {
    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    let array_ptr = std::ptr::without_provenance_mut::<libc::epoll_event>(0x100);
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, epfd, array_ptr) };
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
    assert_eq!(res, -1);
}

// We previously used check_and_update_readiness the moment a file description is registered in an
// epoll instance. But this has an unfortunate side effect of returning notification to another
// epfd that shouldn't receive notification.
fn test_epoll_ctl_notification() {
    // Create an epoll instance.
    let epfd0 = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd0, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Register one side of the socketpair with epoll.
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fds[0] as u64 };
    let res = unsafe { libc::epoll_ctl(epfd0, libc::EPOLL_CTL_ADD, fds[0], &mut ev) };
    assert_eq!(res, 0);

    // epoll_wait to clear notification for epfd0.
    let expected_event = u32::try_from(libc::EPOLLOUT).unwrap();
    let expected_value = fds[0] as u64;
    check_epoll_wait::<1>(epfd0, &[(expected_event, expected_value)]);

    // Create another epoll instance.
    let epfd1 = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd1, -1);

    // Register the same file description for epfd1.
    let mut ev = libc::epoll_event { events: EPOLL_IN_OUT_ET, u64: fds[0] as u64 };
    let res = unsafe { libc::epoll_ctl(epfd1, libc::EPOLL_CTL_ADD, fds[0], &mut ev) };
    assert_eq!(res, 0);
    check_epoll_wait::<1>(epfd1, &[(expected_event, expected_value)]);

    // Previously this epoll_wait will receive a notification, but we shouldn't return notification
    // for this epfd, because there is no I/O event between the two epoll_wait.
    check_epoll_wait::<1>(epfd0, &[]);
}

// Test for ICE caused by weak epoll interest upgrade succeed, but the attempt to retrieve
// the epoll instance based on the epoll file descriptor value failed. EpollEventInterest
// should store a WeakFileDescriptionRef instead of the file descriptor number, so if the
// epoll instance is duped, it'd still be usable after `close` is called on the original
// epoll file descriptor.
// https://github.com/rust-lang/miri/issues/3858
fn test_issue_3858() {
    // Create an eventfd instance.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };

    // Create an epoll instance.
    let epfd = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd, -1);

    // Register eventfd with EPOLLIN | EPOLLET.
    let mut ev = libc::epoll_event {
        events: (libc::EPOLLIN | libc::EPOLLET) as _,
        u64: u64::try_from(fd).unwrap(),
    };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
    assert_eq!(res, 0);

    // Dup the epoll instance.
    let newfd = unsafe { libc::dup(epfd) };
    assert_ne!(newfd, -1);

    // Close the old epoll instance, so the new FD is now the only FD.
    let res = unsafe { libc::close(epfd) };
    assert_eq!(res, 0);

    // Write to the eventfd instance.
    let sized_8_data: [u8; 8] = 1_u64.to_ne_bytes();
    let res = unsafe { libc_utils::write_all(fd, sized_8_data.as_ptr() as *const libc::c_void, 8) };
    assert_eq!(res, 8);
}
