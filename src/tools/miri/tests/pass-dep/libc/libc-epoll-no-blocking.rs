//@only-target: linux android illumos
//@revisions: edge_triggered level_triggered

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::epoll::*;
use libc_utils::*;

/// When the `edge_triggered` revision is active, this is EPOLLET, otherwise
/// it's zero which means we perform level-triggered epolls.
const EPOLLET_OR_ZERO: libc::c_int = if cfg!(edge_triggered) { EPOLLET } else { 0 };

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
    test_epoll_mixed_modes();
    test_epoll_registered_mode_switch();
    test_issue_3858();
    test_issue_4374();
    test_issue_4374_reads();
}

fn test_epoll_socketpair() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Write to fds[0]
    write_all(fds[0], b"abcde").unwrap();

    // Register fds[1] with EPOLLIN|EPOLLOUT|EPOLLRDHUP (and EPOLLET if we're
    // in the `edge_triggered` revision).
    epoll_ctl_add(epfd, fds[1], EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET_OR_ZERO).unwrap();

    // Check result from epoll_wait.
    check_epoll_wait_noblock(epfd, &[Ev { data: fds[1], events: EPOLLIN | EPOLLOUT }]);

    if cfg!(edge_triggered) {
        // Check that this is indeed using "ET" (edge-trigger) semantics: a second wait
        // should return nothing.
        check_epoll_wait_noblock(epfd, &[]);
    } else {
        // Check that this is indeed using "LT" (level-trigger) semantics: a second wait
        // should return the same readiness.
        check_epoll_wait_noblock(epfd, &[Ev { data: fds[1], events: EPOLLIN | EPOLLOUT }]);
    }

    // Write some more to fds[0].
    write_all(fds[0], b"abcde").unwrap();

    // This did not change the readiness of fds[1], so we should get no event.
    // However, Linux seems to always deliver spurious events to the peer on each write,
    // so we match that.
    check_epoll_wait_noblock(epfd, &[Ev { data: fds[1], events: EPOLLIN | EPOLLOUT }]);

    // Close the peer socketpair.
    errno_check(unsafe { libc::close(fds[0]) });

    // Check result from epoll_wait. We expect to get a read, write, HUP notification from the close
    // since closing an FD always unblocks reads and writes on its peer.
    check_epoll_wait_noblock(
        epfd,
        &[Ev { data: fds[1], events: EPOLLIN | EPOLLOUT | EPOLLHUP | EPOLLRDHUP }],
    );
}

// This test first registers a file description with a flag that does not lead to notification,
// then EPOLL_CTL_MOD to add another flag that will lead to notification.
// Also check that the new data value set via MOD is applied properly.
fn test_epoll_ctl_mod() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Register fds[1] with EPOLLIN (and EPOLLET if we're in the `edge_triggered` revision), and data of "0".
    epoll_ctl(epfd, EPOLL_CTL_ADD, fds[1], Ev { events: EPOLLIN | EPOLLET_OR_ZERO, data: 0 })
        .unwrap();

    // Check result from epoll_wait. No notification would be returned.
    check_epoll_wait_noblock(epfd, &[]);

    // Use EPOLL_CTL_MOD to change to EPOLLOUT flag and data.
    epoll_ctl(epfd, EPOLL_CTL_MOD, fds[1], Ev { events: EPOLLOUT | EPOLLET_OR_ZERO, data: 1 })
        .unwrap();

    // Check result from epoll_wait. EPOLLOUT notification and new data is expected.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLOUT, data: 1 }]);

    // Write to fds[1] and read from fds[0] to make the notification ready again
    // (relying on there always being an event when the buffer gets emptied).
    write_all(fds[1], "abc".as_bytes()).unwrap();
    read_exact_array::<3>(fds[0]).unwrap();

    // Now that the event is already ready, change the "data" value.
    epoll_ctl(epfd, EPOLL_CTL_MOD, fds[1], Ev { events: EPOLLOUT | EPOLLET_OR_ZERO, data: 2 })
        .unwrap();

    // Receive event, with latest data value.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLOUT, data: 2 }]);

    // Do another update that changes nothing.
    epoll_ctl(epfd, EPOLL_CTL_MOD, fds[1], Ev { events: EPOLLOUT | EPOLLET_OR_ZERO, data: 2 })
        .unwrap();

    // This re-triggers the event, even if it's the same flags as before.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLOUT, data: 2 }]);
}

fn test_epoll_ctl_del() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Write to fds[0]
    libc_utils::write_all(fds[0], b"abcde").unwrap();

    // Register fds[1] with EPOLLIN|EPOLLOUT (and EPOLLET if we're in the `edge_triggered` revision).
    let mut ev = libc::epoll_event {
        events: (EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO) as u32,
        u64: u64::try_from(fds[1]).unwrap(),
    };
    let res = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fds[1], &mut ev) };
    assert_eq!(res, 0);

    // Test EPOLL_CTL_DEL.
    let res = unsafe { libc::epoll_ctl(epfd, EPOLL_CTL_DEL, fds[1], &mut ev) };
    assert_eq!(res, 0);
    check_epoll_wait_noblock(epfd, &[]);
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
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Write to the socketpair.
    libc_utils::write_all(fds[0], b"abcde").unwrap();

    // Register one side of the socketpair with EPOLLIN | EPOLLOUT (and EPOLLET
    // if we're in the `edge_triggered` revision).
    epoll_ctl_add(epfd1, fds[1], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();
    epoll_ctl_add(epfd2, fds[1], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Notification should be received from both instance of epoll.
    check_epoll_wait_noblock(epfd1, &[Ev { events: EPOLLIN | EPOLLOUT, data: fds[1] }]);
    check_epoll_wait_noblock(epfd2, &[Ev { events: EPOLLIN | EPOLLOUT, data: fds[1] }]);
}

// This test is for two same file description registered under the same epoll instance through dup.
// Notification should be provided for both.
fn test_two_same_fd_in_same_epoll_instance() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Dup the fd.
    let newfd = unsafe { libc::dup(fds[1]) };
    assert_ne!(newfd, -1);

    // Register both fd to the same epoll instance.
    epoll_ctl_add(epfd, fds[1], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();
    epoll_ctl_add(epfd, newfd, EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Write to the socketpair.
    libc_utils::write_all(fds[0], b"abcde").unwrap();

    // Two notification should be received.
    check_epoll_wait_noblock(
        epfd,
        &[
            Ev { events: EPOLLIN | EPOLLOUT, data: fds[1] },
            Ev { events: EPOLLIN | EPOLLOUT, data: newfd },
        ],
    );
}

fn test_epoll_eventfd() {
    // Create an eventfd instance.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();

    // Write 1 to the eventfd instance.
    libc_utils::write_all(fd, &1_u64.to_ne_bytes()).unwrap();

    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Register eventfd with EPOLLIN | EPOLLOUT (and EPOLLET if we're in the `edge_triggered`
    // revision).
    epoll_ctl_add(epfd, fd, EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Check result from epoll_wait.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLIN | EPOLLOUT, data: fd }]);

    // Write 0 to the eventfd.
    libc_utils::write_all(fd, &0_u64.to_ne_bytes()).unwrap();

    // This does not change the status, so we should get no event.
    // However, Linux performs a spurious wakeup.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLIN | EPOLLOUT, data: fd }]);

    // Read from the eventfd.
    libc_utils::read_exact_array::<8>(fd).unwrap();

    // This consumes the event, so the read status is gone. However, deactivation
    // does not trigger an event.
    // Still, we see a spurious wakeup.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLOUT, data: fd }]);

    // Write the maximum possible value.
    libc_utils::write_all(fd, &(u64::MAX - 1).to_ne_bytes()).unwrap();

    // This reactivates reads, therefore triggering an event. Writing is no longer possible.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLIN, data: fd }]);
}

// When read/write happened on one side of the socketpair, only the other side will be notified.
fn test_epoll_socketpair_both_sides() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Register both fd to the same epoll instance.
    epoll_ctl_add(epfd, fds[0], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();
    epoll_ctl_add(epfd, fds[1], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Write to fds[1].
    // (We do the write after the register here, unlike in `test_epoll_socketpair`, to ensure
    // we cover both orders in which this could be done.)
    libc_utils::write_all(fds[1], b"abcde").unwrap();

    // Two notification should be received.
    check_epoll_wait_noblock(
        epfd,
        &[Ev { events: EPOLLIN | EPOLLOUT, data: fds[0] }, Ev { events: EPOLLOUT, data: fds[1] }],
    );

    // Read from fds[0].
    let buf = libc_utils::read_exact_array::<5>(fds[0]).unwrap();
    assert_eq!(buf, *b"abcde");

    if cfg!(edge_triggered) {
        // The state of fds[1] does not change (was writable, is writable).
        // However, we force a spurious wakeup as the read buffer just got emptied.
        // fds[0] lost its readability, but becoming less active is not considered an "edge".
        check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLOUT, data: fds[1] }])
    } else {
        // With level-triggered epoll, only the readable readiness for fds[0] should
        // no longer be reported. The rest stays the same.
        check_epoll_wait_noblock(
            epfd,
            &[Ev { events: EPOLLOUT, data: fds[0] }, Ev { events: EPOLLOUT, data: fds[1] }],
        );
    }
}

// When file description is fully closed, epoll_wait should not provide any notification for
// that file description.
fn test_closed_fd() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create an eventfd instance.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();

    // Register eventfd with EPOLLIN | EPOLLOUT (and EPOLLET if we're in the `edge_triggered` revision).
    epoll_ctl_add(epfd, fd, EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Write to the eventfd instance.
    libc_utils::write_all(fd, &1_u64.to_ne_bytes()).unwrap();

    // Close the eventfd.
    errno_check(unsafe { libc::close(fd) });

    // No notification should be provided because the file description is closed.
    check_epoll_wait_noblock(epfd, &[]);
}

// When a certain file descriptor registered with epoll is closed, but the underlying file description
// is not closed, notification should still be provided.
//
// This is a quirk of epoll being described in https://man7.org/linux/man-pages/man7/epoll.7.html
// A file descriptor is removed from an interest list only after all the file descriptors
// referring to the underlying open file description have been closed.
fn test_not_fully_closed_fd() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create an eventfd instance.
    let fd =
        errno_result(unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) }).unwrap();

    // Dup the fd.
    let newfd = errno_result(unsafe { libc::dup(fd) }).unwrap();

    // Register eventfd with EPOLLIN | EPOLLOUT (and EPOLLET if we're in the `edge_triggered` revision).
    epoll_ctl_add(epfd, fd, EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Close the original fd that being used to register with epoll.
    errno_check(unsafe { libc::close(fd) });

    // Notification should still be provided because the file description is not closed.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLOUT, data: fd }]);

    // Write to the eventfd instance to produce notification.
    libc_utils::write_all(newfd, &1_u64.to_ne_bytes()).unwrap();

    // Close the dupped fd.
    errno_check(unsafe { libc::close(newfd) });

    // No notification should be provided.
    check_epoll_wait_noblock(epfd, &[]);
}

// Each time a notification is provided, it should reflect the file description's readiness
// at the moment the latest event occurred.
fn test_event_overwrite() {
    // Create an eventfd instance.
    let fd =
        errno_result(unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) }).unwrap();

    // Write to the eventfd instance.
    libc_utils::write_all(fd, &1_u64.to_ne_bytes()).unwrap();

    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Register eventfd with EPOLLIN | EPOLLOUT (and EPOLLET if we're in the `edge_triggered` revision).
    epoll_ctl_add(epfd, fd, EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Read from the eventfd instance.
    let mut buf: [u8; 8] = [0; 8];
    let res = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), 8) };
    assert_eq!(res, 8);

    // Check result from epoll_wait.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLOUT, data: fd }]);
}

// An epoll notification will be provided for every succesful read in a socketpair.
// This behaviour differs from the real system.
fn test_socketpair_read() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Register both fd to the same epoll instance.
    epoll_ctl_add(epfd, fds[0], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();
    epoll_ctl_add(epfd, fds[1], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Write a bunch of data bytes to fds[1].
    let data = [42u8; 1024];
    libc_utils::write_all(fds[1], &data).unwrap();

    // Two notification should be received.
    check_epoll_wait_noblock(
        epfd,
        &[Ev { events: EPOLLIN | EPOLLOUT, data: fds[0] }, Ev { events: EPOLLOUT, data: fds[1] }],
    );

    // Read some of the data from fds[0].
    let mut buf = [0; 512];
    libc_utils::read_exact(fds[0], &mut buf).unwrap();
    if cfg!(edge_triggered) {
        // fds[1] did not change, it is still writable, so we get no event
        // in edge-triggered mode.
        check_epoll_wait_noblock(epfd, &[]);
    } else {
        // In level-triggered mode we expect the same events as before because
        // we didn't read everything in the buffer.
        check_epoll_wait_noblock(
            epfd,
            &[
                Ev { events: EPOLLIN | EPOLLOUT, data: fds[0] },
                Ev { events: EPOLLOUT, data: fds[1] },
            ],
        );
    }

    // Read until the buffer is empty.
    let rest = data.len() - buf.len();
    libc_utils::read_exact(fds[0], &mut buf[..rest]).unwrap();

    if cfg!(edge_triggered) {
        // Now we get a notification that fds[1] can be written. This is spurious since it
        // could already be written before, but Linux seems to always emit a notification for
        // the writer when a read empties the buffer.
        check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLOUT, data: fds[1] }]);
    } else {
        // In level-triggered mode we expect the same events as before just without
        // the readable readiness of fds[0] because we now read everything.
        check_epoll_wait_noblock(
            epfd,
            &[Ev { events: EPOLLOUT, data: fds[0] }, Ev { events: EPOLLOUT, data: fds[1] }],
        );
    }
}

// This is to test whether a flag that we don't register won't trigger notification.
fn test_no_notification_for_unregister_flag() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Register fds[0] with EPOLLOUT (and EPOLLET when we're in the `edge_triggered` revision).
    epoll_ctl_add(epfd, fds[0], EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Write to fds[1].
    libc_utils::write_all(fds[1], b"abcde").unwrap();

    // Check result from epoll_wait. Since we didn't register EPOLLIN flag, the notification won't
    // contain EPOLLIN even though fds[0] is now readable.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLOUT, data: fds[0] }]);
}

fn test_epoll_wait_maxevent_zero() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();
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
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Write to fds[0]
    libc_utils::write_all(fds[0], b"abcde").unwrap();

    // Close fds[1].
    // EPOLLERR will be triggered if we close peer fd that still has data in its read buffer.
    errno_check(unsafe { libc::close(fds[1]) });

    // Register fds[1] with EPOLLIN|EPOLLOUT|EPOLLRDHUP (and EPOLLET when we're in the
    // `edge_triggered` revision).
    epoll_ctl_add(epfd, fds[0], EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET_OR_ZERO).unwrap();

    // Check result from epoll_wait.
    check_epoll_wait_noblock(
        epfd,
        &[Ev { events: EPOLLIN | EPOLLOUT | EPOLLHUP | EPOLLRDHUP | EPOLLERR, data: fds[0] }],
    );
}

// This is a test for https://github.com/rust-lang/miri/issues/3812,
// epoll can lose events if they don't fit in the output buffer.
fn test_epoll_lost_events() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Register both fd to the same epoll instance.
    epoll_ctl_add(epfd, fds[0], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();
    epoll_ctl_add(epfd, fds[1], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Two notification should be received. But we only provide buffer for one event.
    check_epoll_wait_explicit(epfd, &[Ev { events: EPOLLOUT, data: fds[0] }], 1, 0);

    if cfg!(edge_triggered) {
        // Previous event should be returned for the second epoll_wait but because we're
        // edge-triggered the first event should no longer be returned.
        check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLOUT, data: fds[1] }]);
    } else {
        // Both events should be returned in level-triggered mode when
        // we provide a big enough buffer.
        check_epoll_wait_noblock(
            epfd,
            &[Ev { events: EPOLLOUT, data: fds[1] }, Ev { events: EPOLLOUT, data: fds[0] }],
        );
    }
}

// This is testing if closing an fd that is already in ready list will cause an empty entry in
// returned notification.
// Related discussion in https://github.com/rust-lang/miri/pull/3818#discussion_r1720679440.
fn test_ready_list_fetching_logic() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create two eventfd instances.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd0 = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();
    let fd1 = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();

    // Register both fd to the same epoll instance. At this point, both of them are on the ready list.
    epoll_ctl_add(epfd, fd0, EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();
    epoll_ctl_add(epfd, fd1, EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Close fd0 so the first entry in the ready list will be empty.
    errno_check(unsafe { libc::close(fd0) });

    // Notification for fd1 should be returned.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLOUT, data: fd1 }]);
}

// In epoll_ctl, if the value of epfd equals to fd, EFAULT should be returned.
// (The docs say loops cause EINVAL, but experiments show it is EFAULT.)
fn test_epoll_ctl_epfd_equal_fd() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    let array_ptr = std::ptr::without_provenance_mut::<libc::epoll_event>(0x100);
    let res = unsafe { libc::epoll_ctl(epfd, EPOLL_CTL_ADD, epfd, array_ptr) };
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::EFAULT));
    assert_eq!(res, -1);
}

// We previously used check_and_update_readiness the moment a file description is registered in an
// epoll instance. But this has an unfortunate side effect of returning notification to another
// epfd that shouldn't receive a notification in edge-triggered mode.
fn test_epoll_ctl_notification() {
    // Create an epoll instance.
    let epfd0 = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd0, -1);

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Register one side of the socketpair with epoll.
    epoll_ctl_add(epfd0, fds[0], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // epoll_wait to clear notification for epfd0.
    check_epoll_wait_noblock(epfd0, &[Ev { events: EPOLLOUT, data: fds[0] }]);

    // Create another epoll instance.
    let epfd1 = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd1, -1);

    // Register the same file description for epfd1.
    epoll_ctl_add(epfd1, fds[0], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();
    check_epoll_wait_noblock(epfd1, &[Ev { events: EPOLLOUT, data: fds[0] }]);

    if cfg!(edge_triggered) {
        // Previously this epoll_wait will receive a notification, but we shouldn't return notification
        // for this epfd, because there is no I/O event between the two epoll_wait.
        check_epoll_wait_noblock(epfd0, &[]);
    } else {
        // We should still get the same events in level-triggered mode.
        check_epoll_wait_noblock(epfd0, &[Ev { events: EPOLLOUT, data: fds[0] }]);
    }
}

/// Test storing a level-triggered and an edge-triggered file descriptor
/// in the same epoll instance and calling `epoll_wait` multiple times.
fn test_epoll_mixed_modes() {
    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Create a socketpair instance.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Register both fd to the same epoll instance.
    // `fds[0]` is added in edge-triggered mode whilst `fds[1]` is added in level-triggered mode.
    epoll_ctl_add(epfd, fds[0], EPOLLIN | EPOLLOUT | EPOLLET).unwrap();
    epoll_ctl_add(epfd, fds[1], EPOLLIN | EPOLLOUT | 0).unwrap();

    // Write to `fds[1]`.
    libc_utils::write_all(fds[1], b"abcde").unwrap();

    // Two events should be received.
    check_epoll_wait_noblock(
        epfd,
        &[Ev { events: EPOLLIN | EPOLLOUT, data: fds[0] }, Ev { events: EPOLLOUT, data: fds[1] }],
    );

    // If we call epoll_wait again immediately, only the level-triggered interests should be received again.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLOUT, data: fds[1] }]);
}

/// Test first registering a file descriptor in edge-triggered mode,
/// then consuming it's readiness and then changing it to level-triggered
/// mode.
fn test_epoll_registered_mode_switch() {
    // Create an eventfd instance.
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();

    // Write 1 to the eventfd instance.
    libc_utils::write_all(fd, &1_u64.to_ne_bytes()).unwrap();

    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Register eventfd with EPOLLIN | EPOLLOUT | EPOLLET.
    epoll_ctl_add(epfd, fd, EPOLLIN | EPOLLOUT | EPOLLET).unwrap();

    // Check result from `epoll_wait`.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLIN | EPOLLOUT, data: fd }]);

    // Because `fd` is registered in edge-triggered mode, the next `epoll_wait` shouldn't
    // return any events.
    check_epoll_wait_noblock(epfd, &[]);

    // Update the registration for `fd` to switch to level-triggered mode.
    epoll_ctl(epfd, EPOLL_CTL_MOD, fd, Ev { events: EPOLLIN | EPOLLOUT, data: fd }).unwrap();

    // Because `fd` is now registered in level-triggered mode, we should see
    // the same events as from the first `epoll_wait`.
    check_epoll_wait_noblock(epfd, &[Ev { events: EPOLLIN | EPOLLOUT, data: fd }]);
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
    let fd = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();

    // Create an epoll instance.
    let epfd = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Register eventfd with EPOLLIN (and EPOLLET if we're in the `edge_triggered` revision).
    epoll_ctl_add(epfd, fd, EPOLLIN | EPOLLET_OR_ZERO).unwrap();

    // Dup the epoll instance.
    let newfd = unsafe { libc::dup(epfd) };
    assert_ne!(newfd, -1);

    // Close the old epoll instance, so the new FD is now the only FD.
    errno_check(unsafe { libc::close(epfd) });

    // Write to the eventfd instance.
    libc_utils::write_all(fd, &1_u64.to_ne_bytes()).unwrap();
}

/// Ensure that if a socket becomes un-writable, we don't see it any more.
fn test_issue_4374() {
    // Create an epoll instance.
    let epfd0 = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd0, -1);

    // Create a socketpair instance, make it non-blocking.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });
    assert_eq!(unsafe { libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK) }, 0);
    assert_eq!(unsafe { libc::fcntl(fds[1], libc::F_SETFL, libc::O_NONBLOCK) }, 0);

    // Register fds[0] with epoll while it is writable (but not readable).
    epoll_ctl_add(epfd0, fds[0], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Fill up fds[0] so that it is not writable any more.
    let zeros = [0u8; 512];
    loop {
        let res = libc_utils::write_all(fds[0], &zeros);
        if let Err(err) = res {
            assert_eq!(err.kind(), std::io::ErrorKind::WouldBlock);
            break;
        }
    }

    // This should have canceled the previous readiness, so now we get nothing.
    check_epoll_wait_noblock(epfd0, &[]);
}

/// Same as above, but for becoming un-readable.
fn test_issue_4374_reads() {
    // Create an epoll instance.
    let epfd0 = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd0, -1);

    // Create a socketpair instance, make it non-blocking.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });
    assert_eq!(unsafe { libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK) }, 0);
    assert_eq!(unsafe { libc::fcntl(fds[1], libc::F_SETFL, libc::O_NONBLOCK) }, 0);

    // Write to fds[1] so that fds[0] becomes readable.
    libc_utils::write_all(fds[1], b"abcde").unwrap();

    // Register fds[0] with epoll while it is readable.
    epoll_ctl_add(epfd0, fds[0], EPOLLIN | EPOLLOUT | EPOLLET_OR_ZERO).unwrap();

    // Read fds[0] so it is no longer readable.
    libc_utils::read_exact_array::<5>(fds[0]).unwrap();

    // We should now still see a notification, but only about it being writable.
    check_epoll_wait_noblock(epfd0, &[Ev { events: EPOLLOUT, data: fds[0] }]);
}
