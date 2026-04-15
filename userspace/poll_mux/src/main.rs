#![no_std]
#![no_main]
extern crate alloc;

use abi::syscall::{poll_flags, PollFd};
use stem::syscall::vfs::*;

#[stem::main]
fn main(_arg: usize) -> ! {
    stem::println!("--- poll_mux start ---");

    // 1. Create a pipe
    let mut pipefds = [0u32; 2];
    pipe(&mut pipefds).expect("pipe failed");
    let (pr, pw) = (pipefds[0], pipefds[1]);
    stem::println!("Pipe created: read={}, write={}", pr, pw);

    // 2. Create a channel and bridge it to a VFS fd
    let (c_write, c_read) = stem::syscall::channel_create(1024).expect("channel create failed");
    let c_read_fd = vfs_fd_from_handle(c_read).expect("vfs_fd_from_handle failed");
    stem::println!(
        "Channel created: write={}, read={}, bridged_fd={}",
        c_write,
        c_read,
        c_read_fd
    );

    // 3. Open a regular device file (/dev/null is always present).
    //    Regular VFS files report POLLIN|POLLOUT immediately (they are
    //    always-ready, matching POSIX semantics for non-socket file descriptors).
    let dev_null_fd =
        vfs_open("/dev/null", abi::syscall::vfs_flags::O_RDWR).expect("open /dev/null failed");
    stem::println!("VFS file opened: fd={}", dev_null_fd);

    // 4. Test timeout — pipe and channel are both empty so poll should expire.
    let mut fds = [
        PollFd {
            fd: pr as i32,
            events: poll_flags::POLLIN,
            revents: 0,
        },
        PollFd {
            fd: c_read_fd as i32,
            events: poll_flags::POLLIN,
            revents: 0,
        },
    ];
    stem::println!("Polling pipe+channel for 100ms (should timeout)...");
    let start = stem::syscall::monotonic_ns();
    let n = vfs_poll(&mut fds, 100).expect("poll failed");
    let end = stem::syscall::monotonic_ns();
    stem::println!(
        "Poll returned {} entries, took {} ms",
        n,
        (end - start) / 1_000_000
    );
    assert!(n == 0, "Expected timeout, got {}", n);

    // 5. VFS regular files are always POLLIN-ready.
    let mut vfs_fds = [PollFd {
        fd: dev_null_fd as i32,
        events: poll_flags::POLLIN | poll_flags::POLLOUT,
        revents: 0,
    }];
    stem::println!("Polling VFS file (should be immediately ready)...");
    let n = vfs_poll(&mut vfs_fds, 0).expect("poll vfs failed");
    stem::println!("Poll returned {} entries", n);
    assert!(n == 1, "Expected 1 ready entry for VFS file, got {}", n);
    assert!(
        vfs_fds[0].revents & poll_flags::POLLIN != 0,
        "Expected POLLIN on VFS file"
    );
    assert!(
        vfs_fds[0].revents & poll_flags::POLLOUT != 0,
        "Expected POLLOUT on VFS file"
    );

    // 6. Test pipe readiness (write before poll)
    vfs_write(pw, b"hello").expect("write to pipe failed");
    stem::println!("Wrote to pipe, polling now...");
    fds[0].revents = 0;
    fds[1].revents = 0;
    let n = vfs_poll(&mut fds, 100).expect("poll failed");
    stem::println!("Poll returned {} entries", n);
    assert!(n == 1, "Expected 1 ready entry, got {}", n);
    assert!(
        fds[0].revents & poll_flags::POLLIN != 0,
        "Expected POLLIN on pipe"
    );

    // 7. Test channel readiness (write before poll)
    stem::syscall::channel_send(c_write, b"world").expect("send to channel failed");
    stem::println!("Sent to channel, polling now...");
    fds[0].revents = 0;
    fds[1].revents = 0;
    let n = vfs_poll(&mut fds, 100).expect("poll failed");
    stem::println!("Poll returned {} entries", n);
    // Both should be ready now if order is preserved
    assert!(
        fds[0].revents & poll_flags::POLLIN != 0,
        "Expected POLLIN on pipe"
    );
    assert!(
        fds[1].revents & poll_flags::POLLIN != 0,
        "Expected POLLIN on channel"
    );

    // 8. Mixed poll: channel + pipe + VFS file all at once.
    //    Pipe (index 0) and channel (index 1) still have unread data;
    //    VFS file (index 2) is always ready.  All three should fire.
    stem::println!("Mixed poll: pipe + channel + VFS file...");
    let mut mixed = [
        PollFd {
            fd: pr as i32,
            events: poll_flags::POLLIN,
            revents: 0,
        },
        PollFd {
            fd: c_read_fd as i32,
            events: poll_flags::POLLIN,
            revents: 0,
        },
        PollFd {
            fd: dev_null_fd as i32,
            events: poll_flags::POLLIN | poll_flags::POLLOUT,
            revents: 0,
        },
    ];
    let n = vfs_poll(&mut mixed, 0).expect("mixed poll failed");
    stem::println!("Mixed poll returned {} entries", n);
    assert!(
        n == 3,
        "Expected all 3 fds ready (pipe + channel + VFS file), got {}",
        n
    );
    assert!(
        mixed[0].revents & poll_flags::POLLIN != 0,
        "Expected POLLIN on pipe in mixed"
    );
    assert!(
        mixed[1].revents & poll_flags::POLLIN != 0,
        "Expected POLLIN on channel in mixed"
    );
    assert!(
        mixed[2].revents & (poll_flags::POLLIN | poll_flags::POLLOUT) != 0,
        "Expected readiness on VFS file in mixed"
    );

    stem::println!("--- poll_mux success ---");
    stem::syscall::exit(0)
}
