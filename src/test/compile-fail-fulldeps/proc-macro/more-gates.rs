// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:more-gates.rs

#![feature(proc_macro)]

extern crate more_gates as foo;

use foo::*;

#[attr2mod]
//~^ ERROR: cannot expand to modules
pub fn a() {}
#[attr2mac1]
//~^ ERROR: cannot expand to macro definitions
pub fn a() {}
#[attr2mac2]
//~^ ERROR: cannot expand to macro definitions
pub fn a() {}

mac2mod!(); //~ ERROR: cannot expand to modules
mac2mac1!(); //~ ERROR: cannot expand to macro definitions
mac2mac2!(); //~ ERROR: cannot expand to macro definitions

tricky!();
//~^ ERROR: cannot expand to modules
//~| ERROR: cannot expand to macro definitions

fn main() {}
