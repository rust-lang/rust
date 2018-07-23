// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(stable_features)]
#![deny(duplicate_features)]

#![feature(rust1)]
#![feature(rust1)] //~ ERROR duplicate `rust1` feature attribute

#![feature(if_let)]
#![feature(if_let)] //~ ERROR duplicate `if_let` feature attribute

fn main() {}
