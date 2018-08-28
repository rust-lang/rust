// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(tuple_struct_self_ctor)]

#![allow(dead_code)]

use std::fmt::Display;

struct ST1(i32, i32);

impl ST1 {
    fn new() -> Self {
        ST1(0, 1)
    }

    fn ctor() -> Self {
        Self(1,2)         // Self as a constructor
    }

    fn pattern(self) {
        match self {
            Self(x, y) => println!("{} {}", x, y), // Self as a pattern
        }
    }
}

struct ST2<T>(T); // With type parameter

impl<T> ST2<T> where T: Display {

    fn ctor(v: T) -> Self {
        Self(v)
    }

    fn pattern(&self) {
        match self {
            Self(ref v) => println!("{}", v),
        }
    }
}

struct ST3<'a>(&'a i32); // With lifetime parameter

impl<'a> ST3<'a> {

    fn ctor(v: &'a i32) -> Self {
        Self(v)
    }

    fn pattern(self) {
        let Self(ref v) = self;
        println!("{}", v);
    }
}

struct ST4(usize);

impl ST4 {
    fn map(opt: Option<usize>) -> Option<Self> {
        opt.map(Self)     // use `Self` as a function passed somewhere
    }
}

struct ST5;               // unit struct

impl ST5 {
    fn ctor() -> Self {
        Self               // `Self` as a unit struct value
    }

    fn pattern(self) -> Self {
        match self {
            Self => Self,   // `Self` as a unit struct value for matching
        }
    }
}

fn main() {
    let v1 = ST1::ctor();
    v1.pattern();

    let v2 = ST2::ctor(10);
    v2.pattern();

    let local = 42;
    let v3 = ST3::ctor(&local);
    v3.pattern();

    let v4 = Some(1usize);
    let _ = ST4::map(v4);

    let v5 = ST5::ctor();
    v5.pattern();
}
