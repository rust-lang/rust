// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]
#![feature(in_band_lifetimes)]

struct Foo<'a, 'b> {
    x: &'a u32,
    y: &'b u32,
}

struct Bar<'b> {
    z: &'b u32
}

impl Foo<'_, '_> {
    fn take_bar(&mut self, b: Bar<'_>) {
        self.y = b.z
        //~^ ERROR unsatisfied lifetime constraints
    }
}

fn main() { }
