// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast this is executing itself
#[no_uv];

extern crate native;

use std::os;
use std::io::process::{Process, ProcessConfig};
use std::unstable::finally::Finally;
use std::str;

#[start]
fn start(argc: int, argv: **u8) -> int { native::start(argc, argv, main) }

#[inline(never)]
fn foo() {
    fail!()
}

#[inline(never)]
fn double() {
    (|| {
        fail!("once");
    }).finally(|| {
        fail!("twice");
    })
}

fn runtest(me: &str) {
    let mut env = os::env();
    match env.iter().position(|&(ref s, _)| "RUST_BACKTRACE" == *s) {
        Some(i) => { env.remove(i); }
        None => {}
    }
    env.push((~"RUST_BACKTRACE", ~"1"));

    // Make sure that the stack trace is printed
    let mut p = Process::configure(ProcessConfig {
        program: me,
        args: [~"fail"],
        env: Some(env.as_slice()),
        .. ProcessConfig::new()
    }).unwrap();
    let out = p.wait_with_output();
    assert!(!out.status.success());
    let s = str::from_utf8(out.error).unwrap();
    assert!(s.contains("stack backtrace") && s.contains("foo::h"),
            "bad output: {}", s);

    // Make sure the stack trace is *not* printed
    let mut p = Process::configure(ProcessConfig {
        program: me,
        args: [~"fail"],
        .. ProcessConfig::new()
    }).unwrap();
    let out = p.wait_with_output();
    assert!(!out.status.success());
    let s = str::from_utf8(out.error).unwrap();
    assert!(!s.contains("stack backtrace") && !s.contains("foo::h"),
            "bad output2: {}", s);

    // Make sure a stack trace is printed
    let mut p = Process::configure(ProcessConfig {
        program: me,
        args: [~"double-fail"],
        .. ProcessConfig::new()
    }).unwrap();
    let out = p.wait_with_output();
    assert!(!out.status.success());
    let s = str::from_utf8(out.error).unwrap();
    assert!(s.contains("stack backtrace") && s.contains("double::h"),
            "bad output3: {}", s);

    // Make sure a stack trace isn't printed too many times
    let mut p = Process::configure(ProcessConfig {
        program: me,
        args: [~"double-fail"],
        env: Some(env.as_slice()),
        .. ProcessConfig::new()
    }).unwrap();
    let out = p.wait_with_output();
    assert!(!out.status.success());
    let s = str::from_utf8(out.error).unwrap();
    let mut i = 0;
    for _ in range(0, 2) {
        i += s.slice_from(i + 10).find_str("stack backtrace").unwrap() + 10;
    }
    assert!(s.slice_from(i + 10).find_str("stack backtrace").is_none(),
            "bad output4: {}", s);
}

fn main() {
    let args = os::args();
    if args.len() >= 2 && args[1].as_slice() == "fail" {
        foo();
    } else if args.len() >= 2 && args[1].as_slice() == "double-fail" {
        double();
    } else {
        runtest(args[0]);
    }
}
