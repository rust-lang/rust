//@ run-pass
//@ ignore-android
//@ ignore-fuchsia no '/bin/sh', '/bin/ls'
//@ ignore-tvos `Command::uid/gid` requires fork, which is prohibited
//@ ignore-watchos `Command::uid/gid` requires fork, which is prohibited
//@ needs-subprocess

#![feature(rustc_private)]

fn main() {
    #[cfg(unix)]
    run()
}

#[cfg(unix)]
fn run() {
    extern crate libc;
    use std::process::Command;
    use std::os::unix::prelude::*;

    let mut p = Command::new("/bin/sh")
        .arg("-c").arg("true")
        .uid(unsafe { libc::getuid() })
        .gid(unsafe { libc::getgid() })
        .spawn().unwrap();
    assert!(p.wait().unwrap().success());

    // if we're already root, this isn't a valid test. Most of the bots run
    // as non-root though (android is an exception).
    if unsafe { libc::getuid() != 0 } {
        assert!(Command::new("/bin/ls").uid(0).gid(0).spawn().is_err());
    }
}
