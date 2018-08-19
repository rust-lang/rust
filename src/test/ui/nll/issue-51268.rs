// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![feature(nll)]

struct Bar;

impl Bar {
    fn bar(&mut self, _: impl Fn()) {}
}

struct Foo {
    thing: Bar,
    number: usize,
}

impl Foo {
    fn foo(&mut self) {
        self.thing.bar(|| {
        //~^ ERROR cannot borrow `self.thing` as mutable because it is also borrowed as immutable [E0502]
            &self.number;
        });
    }
}

fn main() {}
