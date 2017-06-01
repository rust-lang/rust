// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
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
    const NAME: &'static str;
}


impl<'a> Foo for &'a () {
//~^ NOTE the lifetime 'a as defined
    const NAME: &'a str = "unit";
    //~^ ERROR mismatched types [E0308]
    //~| NOTE lifetime mismatch
    //~| NOTE expected type `&'static str`
    //~| NOTE ...does not necessarily outlive the static lifetime
}

fn main() {}
