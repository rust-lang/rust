// Verifies that std provides replacement for the standard file descriptors when they are missing.
//
// run-pass
// ignore-windows unix specific test
// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-sgx no processes

#![feature(rustc_private)]
extern crate libc;

use std::io::{self, Read};
use std::os::unix::process::CommandExt;
use std::process::Command;

fn main() {
    let mut args = std::env::args();
    let argv0 = args.next().expect("argv0");
    match args.next().as_deref() {
        Some("child") => child(),
        None => parent(&argv0),
        _ => unreachable!(),
    }
}

fn parent(argv0: &str) {
    let status = unsafe { Command::new(argv0)
        .arg("child")
        .pre_exec(close_std_fds_on_exec)
        .status()
        .expect("failed to execute child process")
    };
    if !status.success() {
        panic!("child failed with {}", status);
    }
}

fn close_std_fds_on_exec() -> io::Result<()> {
    for fd in 0..3 {
        if unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) == -1 } {
            return Err(io::Error::last_os_error())
        }
    }
    Ok(())
}

fn child() {
    // Standard file descriptors should be valid.
    assert_fd_is_valid(0);
    assert_fd_is_valid(1);
    assert_fd_is_valid(2);

    // Writing to stdout & stderr should not panic.
    println!("a");
    println!("b");
    eprintln!("c");
    eprintln!("d");

    // Stdin should be at EOF.
    let mut buffer = Vec::new();
    let n = io::stdin().read_to_end(&mut buffer).unwrap();
    assert_eq!(n, 0);
}

fn assert_fd_is_valid(fd: libc::c_int) {
    if unsafe { libc::fcntl(fd, libc::F_GETFD) == -1 } {
        panic!("file descriptor {} is not valid {}", fd, io::Error::last_os_error());
    }
}
