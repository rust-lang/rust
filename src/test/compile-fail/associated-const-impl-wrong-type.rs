// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_consts)]

trait Foo {
    const BAR: u32; //~ NOTE type in trait
}

struct SignedBar;

impl Foo for SignedBar {
    const BAR: i32 = -1;
    //~^ ERROR implemented const `BAR` has an incompatible type for trait [E0326]
    //~| NOTE expected u32, found i32
}

fn main() {}
