// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const A: usize = { 1; 2 };
//~^ ERROR statements in constants are unstable

const B: usize = { { } 2 };
//~^ ERROR statements in constants are unstable

macro_rules! foo {
    () => (()) //~ ERROR statements in constants are unstable
}
const C: usize = { foo!(); 2 };

const D: usize = { let x = 4; 2 };
//~^ ERROR let bindings in constants are unstable
//~| ERROR statements in constants are unstable
//~| ERROR let bindings in constants are unstable
//~| ERROR statements in constants are unstable

pub fn main() {}
