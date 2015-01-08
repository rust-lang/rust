// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lint-unused-extern-crate.rs

#![deny(unused_extern_crates)]
#![allow(unused_variables)]

extern crate libc; //~ ERROR: unused extern crate

extern crate "collections" as collecs; // no error, it is used

extern crate rand; // no error, the use marks it as used
                   // even if imported objects aren't used

extern crate "lint-unused-extern-crate" as other; // no error, the use * marks it as used

#[allow(unused_imports)]
use rand::isaac::IsaacRng;

use other::*;

fn main() {
    let x: collecs::vec::Vec<usize> = Vec::new();
    let y = foo();
}
