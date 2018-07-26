// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(rustc_attrs)]
#![allow(dead_code)]

fn main() { #![rustc_error] // rust-lang/rust#49855
    let mut x = 33;

    let p = &x;
    x = 22; //~ ERROR cannot assign to `x` because it is borrowed [E0506]
}
