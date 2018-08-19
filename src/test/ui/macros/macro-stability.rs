// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:unstable-macros.rs

#![feature(staged_api)]
#[macro_use] extern crate unstable_macros;

#[unstable(feature = "local_unstable", issue = "0")]
macro_rules! local_unstable { () => () }

fn main() {
    local_unstable!();
    unstable_macro!(); //~ ERROR: macro unstable_macro! is unstable
}
