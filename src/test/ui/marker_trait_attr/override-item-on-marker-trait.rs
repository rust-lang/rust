// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(marker_trait_attr)]

#[marker]
trait Marker {
    const N: usize = 0;
    fn do_something() {}
}

struct OverrideConst;
impl Marker for OverrideConst {
//~^ ERROR impls for marker traits cannot contain items
    const N: usize = 1;
}

struct OverrideFn;
impl Marker for OverrideFn {
//~^ ERROR impls for marker traits cannot contain items
    fn do_something() {
        println!("Hello world!");
    }
}

fn main() {}
