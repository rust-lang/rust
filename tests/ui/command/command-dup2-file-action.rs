//@ run-pass
//@ only-unix (this is a unix-specific test)
//@ needs-subprocess
//@ ignore-fuchsia no execvp syscall
//@ ignore-tvos execvp is prohibited
//@ ignore-watchos execvp is prohibited

// Test for CommandExt::dup2_file_action: verifies that a CLOEXEC pipe fd
// can be inherited by a child process when dup2_file_action(fd, fd) is used
// to clear the CLOEXEC flag via posix_spawn_file_actions_adddup2.

#![feature(rustc_private, process_file_actions)]

extern crate libc;

use std::env;
use std::os::unix::process::CommandExt;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() >= 3 && args[1] == "child" {
        // Child mode: read from the inherited fd and print what we got
        let fd: libc::c_int = args[2].parse().unwrap();
        let mut buf = [0u8; 64];
        let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut _, buf.len()) };
        assert!(n > 0, "expected to read from inherited fd {}, got {}", fd, n);
        let msg = std::str::from_utf8(&buf[..n as usize]).unwrap();
        assert_eq!(msg, "hello from parent");
        return;
    }

    // Parent mode: create a pipe (fds are CLOEXEC by default via pipe2),
    // write to the write end, then spawn a child with dup2_file_action
    // to clear CLOEXEC on the read end so the child can read it.

    let mut pipe_fds = [0 as libc::c_int; 2];
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        let ret = unsafe { libc::pipe2(pipe_fds.as_mut_ptr(), libc::O_CLOEXEC) };
        assert_eq!(ret, 0, "pipe2 failed");
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        // macOS doesn't have pipe2, use pipe + fcntl
        let ret = unsafe { libc::pipe(pipe_fds.as_mut_ptr()) };
        assert_eq!(ret, 0, "pipe failed");
        unsafe {
            libc::fcntl(pipe_fds[0], libc::F_SETFD, libc::FD_CLOEXEC);
            libc::fcntl(pipe_fds[1], libc::F_SETFD, libc::FD_CLOEXEC);
        }
    }

    let read_fd = pipe_fds[0];
    let write_fd = pipe_fds[1];

    // Verify the read end has CLOEXEC set
    let flags = unsafe { libc::fcntl(read_fd, libc::F_GETFD) };
    assert!(flags & libc::FD_CLOEXEC != 0, "expected CLOEXEC on read fd");

    // Write data to the pipe
    let msg = b"hello from parent";
    let written = unsafe { libc::write(write_fd, msg.as_ptr() as *const _, msg.len()) };
    assert_eq!(written, msg.len() as isize);
    unsafe { libc::close(write_fd) };

    // Spawn child with dup2_file_action(read_fd, read_fd) to clear CLOEXEC
    let me = env::current_exe().unwrap();
    let output = unsafe {
        Command::new(&me)
            .arg("child")
            .arg(read_fd.to_string())
            .dup2_file_action(read_fd, read_fd)
            .output()
            .unwrap()
    };

    unsafe { libc::close(read_fd) };

    assert!(
        output.status.success(),
        "child failed: status={}, stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}
