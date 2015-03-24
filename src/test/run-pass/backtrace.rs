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
// ignore-windows FIXME #13259

#![feature(unboxed_closures)]
#![feature(unsafe_destructor, old_io, collections)]

use std::env;
use std::old_io::process::Command;
use std::str;
use std::ops::{Drop, FnMut, FnOnce};

#[inline(never)]
fn foo() {
    let _v = vec![1, 2, 3];
    if env::var_os("IS_TEST").is_some() {
        panic!()
    }
}

#[inline(never)]
fn double() {
    struct Double;

    impl Drop for Double {
        fn drop(&mut self) { panic!("twice") }
    }

    let _d = Double;

    panic!("once");
}

fn runtest(me: &str) {
    let mut template = Command::new(me);
    template.env("IS_TEST", "1");

    // Make sure that the stack trace is printed
    let p = template.clone().arg("fail").env("RUST_BACKTRACE", "1").spawn().unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(!out.status.success());
    let s = str::from_utf8(&out.error).unwrap();
    assert!(s.contains("stack backtrace") && s.contains("foo::h"),
            "bad output: {}", s);

    // Make sure the stack trace is *not* printed
    // (Remove RUST_BACKTRACE from our own environment, in case developer
    // is running `make check` with it on.)
    let p = template.clone().arg("fail").env_remove("RUST_BACKTRACE").spawn().unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(!out.status.success());
    let s = str::from_utf8(&out.error).unwrap();
    assert!(!s.contains("stack backtrace") && !s.contains("foo::h"),
            "bad output2: {}", s);

    // Make sure a stack trace is printed
    let p = template.clone().arg("double-fail").spawn().unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(!out.status.success());
    let s = str::from_utf8(&out.error).unwrap();
    // loosened the following from double::h to double:: due to
    // spurious failures on mac, 32bit, optimized
    assert!(s.contains("stack backtrace") && s.contains("double::"),
            "bad output3: {}", s);

    // Make sure a stack trace isn't printed too many times
    let p = template.clone().arg("double-fail")
                                .env("RUST_BACKTRACE", "1").spawn().unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(!out.status.success());
    let s = str::from_utf8(&out.error).unwrap();
    let mut i = 0;
    for _ in 0..2 {
        i += s[i + 10..].find_str("stack backtrace").unwrap() + 10;
    }
    assert!(s[i + 10..].find_str("stack backtrace").is_none(),
            "bad output4: {}", s);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "fail" {
        foo();
    } else if args.len() >= 2 && args[1] == "double-fail" {
        double();
    } else {
        runtest(&args[0]);
    }
}
