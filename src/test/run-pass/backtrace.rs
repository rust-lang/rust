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
#![feature(unsafe_destructor)]

use std::os;
use std::io::process::Command;
use std::str;
use std::ops::{Drop, FnMut, FnOnce};

pub trait Finally<T> {
    fn finally<F>(&mut self, dtor: F) -> T where F: FnMut();
}

impl<T, F> Finally<T> for F where F: FnMut() -> T {
    fn finally<G>(&mut self, mut dtor: G) -> T where G: FnMut() {
        try_finally(&mut (), self, |_, f| (*f)(), |_| dtor())
    }
}

pub fn try_finally<T, U, R, F, G>(mutate: &mut T, drop: U, try_fn: F, finally_fn: G) -> R where
    F: FnOnce(&mut T, U) -> R,
    G: FnMut(&mut T),
{
    let f = Finallyalizer {
        mutate: mutate,
        dtor: finally_fn,
    };
    try_fn(&mut *f.mutate, drop)
}

struct Finallyalizer<'a, A:'a, F> where F: FnMut(&mut A) {
    mutate: &'a mut A,
    dtor: F,
}

#[unsafe_destructor]
impl<'a, A, F> Drop for Finallyalizer<'a, A, F> where F: FnMut(&mut A) {
    #[inline]
    fn drop(&mut self) {
        (self.dtor)(self.mutate);
    }
}


#[inline(never)]
fn foo() {
    let _v = vec![1i, 2, 3];
    if os::getenv("IS_TEST").is_some() {
        panic!()
    }
}

#[inline(never)]
fn double() {
    (|&mut:| {
        panic!("once");
    }).finally(|| {
        panic!("twice");
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
    // loosened the following from double::h to double:: due to
    // spurious failures on mac, 32bit, optimized
    assert!(s.contains("stack backtrace") && s.contains("double::"),
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
