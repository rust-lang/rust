// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(overloaded_calls)]

use std::{fmt, ops};

struct Shower<T> {
    x: T
}

impl<T: fmt::Show> ops::Fn<(), ()> for Shower<T> {
    fn call(&self, _args: ()) {
//~^ ERROR `call` has an incompatible type for trait: expected "rust-call" fn, found "Rust" fn
        println!("{}", self.x);
    }
}

fn make_shower<T>(x: T) -> Shower<T> {
    Shower { x: x }
}

pub fn main() {
    let show3 = make_shower(3i);
    show3();
}
