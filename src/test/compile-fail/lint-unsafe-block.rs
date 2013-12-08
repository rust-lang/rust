// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(unused_unsafe)];
#[allow(dead_code)];
#[deny(unsafe_block)];
#[feature(macro_rules)];

unsafe fn allowed() {}

#[allow(unsafe_block)] fn also_allowed() { unsafe {} }

macro_rules! unsafe_in_macro {
    () => {
        unsafe {} //~ ERROR: usage of an `unsafe` block
    }
}

fn main() {
    unsafe {} //~ ERROR: usage of an `unsafe` block

    unsafe_in_macro!()
}
