// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
// Test that inconsistent bounds are used in well-formedness checks
#![feature(trivial_bounds)]

use std::fmt::Debug;

pub fn foo() where Vec<str>: Debug, str: Copy {
    let x = vec![*"1"];
    println!("{:?}", x);
}

fn main() {}
