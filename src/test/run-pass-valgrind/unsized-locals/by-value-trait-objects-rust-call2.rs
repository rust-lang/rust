// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsized_locals)]
#![feature(unboxed_closures)]

pub trait FnOnce<Args> {
    type Output;
    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

struct A;

impl FnOnce<(String, Box<str>)> for A {
    type Output = String;
    extern "rust-call" fn call_once(self, (s1, s2): (String, Box<str>)) -> Self::Output {
        assert_eq!(&s1 as &str, "s1");
        assert_eq!(&s2 as &str, "s2");
        format!("hello")
    }
}

struct B(i32);

impl FnOnce<(String, Box<str>)> for B {
    type Output = String;
    extern "rust-call" fn call_once(self, (s1, s2): (String, Box<str>)) -> Self::Output {
        assert_eq!(&s1 as &str, "s1");
        assert_eq!(&s2 as &str, "s2");
        format!("{}", self.0)
    }
}

struct C(String);

impl FnOnce<(String, Box<str>)> for C {
    type Output = String;
    extern "rust-call" fn call_once(self, (s1, s2): (String, Box<str>)) -> Self::Output {
        assert_eq!(&s1 as &str, "s1");
        assert_eq!(&s2 as &str, "s2");
        self.0
    }
}

struct D(Box<String>);

impl FnOnce<(String, Box<str>)> for D {
    type Output = String;
    extern "rust-call" fn call_once(self, (s1, s2): (String, Box<str>)) -> Self::Output {
        assert_eq!(&s1 as &str, "s1");
        assert_eq!(&s2 as &str, "s2");
        *self.0
    }
}


fn main() {
    let (s1, s2) = (format!("s1"), format!("s2").into_boxed_str());
    let x = *(Box::new(A) as Box<dyn FnOnce<(String, Box<str>), Output = String>>);
    assert_eq!(x.call_once((s1, s2)), format!("hello"));
    let (s1, s2) = (format!("s1"), format!("s2").into_boxed_str());
    let x = *(Box::new(B(42)) as Box<dyn FnOnce<(String, Box<str>), Output = String>>);
    assert_eq!(x.call_once((s1, s2)), format!("42"));
    let (s1, s2) = (format!("s1"), format!("s2").into_boxed_str());
    let x = *(Box::new(C(format!("jumping fox")))
              as Box<dyn FnOnce<(String, Box<str>), Output = String>>);
    assert_eq!(x.call_once((s1, s2)), format!("jumping fox"));
    let (s1, s2) = (format!("s1"), format!("s2").into_boxed_str());
    let x = *(Box::new(D(Box::new(format!("lazy dog"))))
              as Box<dyn FnOnce<(String, Box<str>), Output = String>>);
    assert_eq!(x.call_once((s1, s2)), format!("lazy dog"));
}
