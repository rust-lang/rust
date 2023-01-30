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

    // If the `clone3` syscall is not implemented in the current kernel version it should return an
    // `ENOSYS` error. Docker also blocks the whole syscall inside unprivileged containers, and
    // returns `EPERM` (instead of `ENOSYS`) when a program tries to invoke the syscall. Because of
    // that we need to check for *both* `ENOSYS` and `EPERM`.
    //
    // Note that Docker's behavior is breaking other projects (notably glibc), so they're planning
    // to update their filtering to return `ENOSYS` in a future release:
    //
    //     https://github.com/moby/moby/issues/42680
    //
    err.raw_os_error() != Some(libc::ENOSYS) && err.raw_os_error() != Some(libc::EPERM)
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
