// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten can't run commands

#![feature(libc)]

extern crate libc;

use std::process::{Command, ExitStatus};
use std::env;

#[link(name = "rust_test_helpers", kind = "static")]
extern {
    fn rust_get_null_ptr() -> *mut ::libc::c_char;
}

#[cfg(unix)]
fn check_status(status: std::process::ExitStatus)
{
    use libc;
    use std::os::unix::process::ExitStatusExt;

    assert!(status.signal() == Some(libc::SIGSEGV)
            || status.signal() == Some(libc::SIGBUS));
}

#[cfg(not(unix))]
fn check_status(status: std::process::ExitStatus)
{
    assert!(!status.success());
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "segfault" {
        unsafe { *rust_get_null_ptr() = 1; }; // trigger a segfault
    } else {
        let segfault = Command::new(&args[0]).arg("segfault").output().unwrap();
        let stderr = String::from_utf8_lossy(&segfault.stderr);
        let stdout = String::from_utf8_lossy(&segfault.stdout);
        println!("stdout: {}", stdout);
        println!("stderr: {}", stderr);
        println!("status: {}", segfault.status);
        check_status(segfault.status);
        assert!(!stderr.contains("has overflowed its stack"));
    }
}
