// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-pretty-expanded FIXME #15189
// ignore-win32 FIXME #13259
extern crate native;

use std::os;
use std::io::process::Command;
use std::finally::Finally;
use std::str;

#[start]
fn start(argc: int, argv: *const *const u8) -> int {
    native::start(argc, argv, main)
}

#[inline(never)]
fn foo() {
    let _v = vec![1i, 2, 3];
    if os::getenv("IS_TEST").is_some() {
        fail!()
    }
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
    let mut template = Command::new(me);
    template.env("IS_TEST", "1");

    // Make sure that the stack trace is printed
    let p = template.clone().arg("fail").env("RUST_BACKTRACE", "1").spawn().unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(!out.status.success());
    let s = str::from_utf8(out.error.as_slice()).unwrap();
    assert!(s.contains("stack backtrace") && s.contains("foo::h"),
            "bad output: {}", s);

    // Make sure the stack trace is *not* printed
    let p = template.clone().arg("fail").spawn().unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(!out.status.success());
    let s = str::from_utf8(out.error.as_slice()).unwrap();
    assert!(!s.contains("stack backtrace") && !s.contains("foo::h"),
            "bad output2: {}", s);

    // Make sure a stack trace is printed
    let p = template.clone().arg("double-fail").spawn().unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(!out.status.success());
    let s = str::from_utf8(out.error.as_slice()).unwrap();
    assert!(s.contains("stack backtrace") && s.contains("double::h"),
            "bad output3: {}", s);

    // Make sure a stack trace isn't printed too many times
    let p = template.clone().arg("double-fail")
                                .env("RUST_BACKTRACE", "1").spawn().unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(!out.status.success());
    let s = str::from_utf8(out.error.as_slice()).unwrap();
    let mut i = 0;
    for _ in range(0i, 2) {
        i += s.slice_from(i + 10).find_str("stack backtrace").unwrap() + 10;
    }
    assert!(s.slice_from(i + 10).find_str("stack backtrace").is_none(),
            "bad output4: {}", s);
}

fn main() {
    let args = os::args();
    let args = args.as_slice();
    if args.len() >= 2 && args[1].as_slice() == "fail" {
        foo();
    } else if args.len() >= 2 && args[1].as_slice() == "double-fail" {
        double();
    } else {
        runtest(args[0].as_slice());
    }
}
