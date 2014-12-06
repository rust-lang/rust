// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test

extern crate libc;

use std::io::process::Command;
use std::iter::IteratorExt;

use libc::funcs::posix88::unistd;


// "ps -A -o pid,sid,command" with GNU ps should output something like this:
//   PID   SID COMMAND
//     1     1 /sbin/init
//     2     0 [kthreadd]
//     3     0 [ksoftirqd/0]
// ...
// 12562  9237 ./spawn-failure
// 12563  9237 [spawn-failure] <defunct>
// 12564  9237 [spawn-failure] <defunct>
// ...
// 12592  9237 [spawn-failure] <defunct>
// 12593  9237 ps -A -o pid,sid,command
// 12884 12884 /bin/zsh
// 12922 12922 /bin/zsh
// ...

#[cfg(unix)]
fn find_zombies() {
    // http://man.freebsd.org/ps(1)
    // http://man7.org/linux/man-pages/man1/ps.1.html
    #[cfg(not(target_os = "macos"))]
    const FIELDS: &'static str = "pid,sid,command";

    // https://developer.apple.com/library/mac/documentation/Darwin/
    // Reference/ManPages/man1/ps.1.html
    #[cfg(target_os = "macos")]
    const FIELDS: &'static str = "pid,sess,command";

    let my_sid = unsafe { unistd::getsid(0) };

    let ps_cmd_output = Command::new("ps").args(&["-A", "-o", FIELDS]).output().unwrap();
    let ps_output = String::from_utf8_lossy(ps_cmd_output.output.as_slice());

    let found = ps_output.split('\n').enumerate().any(|(line_no, line)|
        0 < line_no && 0 < line.len() &&
        my_sid == from_str(line.split(' ').filter(|w| 0 < w.len()).nth(1)
            .expect("1st column should be Session ID")
            ).expect("Session ID string into integer") &&
        line.contains("defunct") && {
            println!("Zombie child {}", line);
            true
        }
    );

    assert!( ! found, "Found at least one zombie child");
}

#[cfg(windows)]
fn find_zombies() { }

fn main() {
    let too_long = format!("/NoSuchCommand{:0300}", 0u8);

    for _ in range(0u32, 100) {
        let invalid = Command::new(too_long.as_slice()).spawn();
        assert!(invalid.is_err());
    }

    find_zombies();
}
