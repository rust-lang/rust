// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests for "default" bounds inferred for traits with no bounds list.


trait Foo {}

fn a(_x: Box<Foo+Send>) {
}

fn b(_x: &'static Foo+'static) {
}

fn c(x: Box<Foo+Sync>) {
    a(x); //~ ERROR mismatched types
}

fn d(x: &'static Foo+Sync) {
    b(x); //~ ERROR cannot infer
    //~^ ERROR mismatched types
}

fn main() {}
