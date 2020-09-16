// run-pass
// linux-only - pidfds are a linux-specific concept

#![feature(linux_pidfd)]
use std::os::linux::process::{CommandExt, ChildExt};
use std::process::Command;

fn main() {
    // We don't assert the precise value, since the standard libarary
    // may be opened other file descriptors before our code ran.
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
