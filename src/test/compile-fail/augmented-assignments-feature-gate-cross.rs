// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:augmented_assignments.rs

// Test that the feature gate is needed when using augmented assignments that were overloaded in
// another crate

extern crate augmented_assignments;

use augmented_assignments::Int;

fn main() {
    let mut x = Int(0);
    x += 1;
    //~^ error: overloaded augmented assignments are not stable
    //~| help: add #![feature(augmented_assignments)] to the crate root to enable
}
