// run-pass
// only-linux - pidfds are a linux-specific concept

#![feature(linux_pidfd)]
#![feature(rustc_private)]

extern crate libc;

use std::io::Error;
use std::os::linux::process::{ChildExt, CommandExt};
use std::process::Command;

fn has_clone3() -> bool {
    let res = unsafe { libc::syscall(libc::SYS_clone3, 0, 0) };
    let err = (res == -1)
        .then(|| Error::last_os_error())
        .expect("probe syscall should not succeed");
    err.raw_os_error() != Some(libc::ENOSYS)
}

fn main() {
    // pidfds require the clone3 syscall
    if !has_clone3() {
        return;
    }

    // We don't assert the precise value, since the standard library
    // might have opened other file descriptors before our code runs.
    let _ = Command::new("echo")
        .create_pidfd(true)
        .spawn()
        .unwrap()
        .pidfd().expect("failed to obtain pidfd");

    let _ = Command::new("echo")
        .create_pidfd(false)
        .spawn()
        .unwrap()
        .pidfd().expect_err("pidfd should not have been created when create_pid(false) is set");

    let _ = Command::new("echo")
        .spawn()
        .unwrap()
        .pidfd().expect_err("pidfd should not have been created");
}
