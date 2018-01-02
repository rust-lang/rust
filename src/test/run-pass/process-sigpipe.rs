// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android since the dynamic linker sets a SIGPIPE handler (to do
// a crash report) so inheritance is moot on the entire platform

// libstd ignores SIGPIPE, and other libraries may set signal masks.
// Make sure that these behaviors don't get inherited to children
// spawned via std::process, since they're needed for traditional UNIX
// filter behavior. This test checks that `yes | head` terminates
// (instead of running forever), and that it does not print an error
// message about a broken pipe.

// ignore-cloudabi no subprocesses support
// ignore-emscripten no threads support

use std::process;
use std::thread;

#[cfg(unix)]
fn main() {
    // Just in case `yes` doesn't check for EPIPE...
    thread::spawn(|| {
        thread::sleep_ms(5000);
        process::exit(1);
    });
    let output = process::Command::new("sh").arg("-c").arg("yes | head").output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.len() == 0);
}

#[cfg(not(unix))]
fn main() {
    // Not worried about signal masks on other platforms
}
