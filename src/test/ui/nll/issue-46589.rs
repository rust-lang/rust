// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

struct Foo;

impl Foo {
    fn get_self(&mut self) -> Option<&mut Self> {
        Some(self)
    }

    fn new_self(&mut self) -> &mut Self {
        self
    }

    fn trigger_bug(&mut self) {
        let other = &mut (&mut *self);

        *other = match (*other).get_self() {
            Some(s) => s,
            None => (*other).new_self()
            //~^ ERROR cannot borrow `**other` as mutable more than once at a time [E0499]
        };

        let c = other;
    }
}

fn main() {}
