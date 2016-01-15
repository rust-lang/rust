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

mod m1 {
    pub use ::E::V; //~ WARN variant `V` is private, and cannot be reexported
    //~^ WARNING hard error
}

mod m2 {
    pub use ::E::{V}; //~ WARN variant `V` is private, and cannot be reexported
    //~^ WARNING hard error
}

mod m3 {
    pub use ::E::V::{self}; //~ WARN variant `V` is private, and cannot be reexported
    //~^ WARNING hard error
}

mod m4 {
    pub use ::E::*; //~ WARN variant `V` is private, and cannot be reexported
    //~^ WARNING hard error
}

enum E { V }

#[rustc_error]
fn main() {} //~ ERROR compilation successful
