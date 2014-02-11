// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test various ways to construct a pointer with a longer lifetime
// than the thing it points at and ensure that they result in
// errors. See also regions-free-region-ordering-callee.rs

fn call1<'a>(x: &'a uint) {
    // Test that creating a pointer like
    // &'a &'z uint requires that 'a <= 'z:
    let y: uint = 3;
    let z: &'a & uint = &(&y);
    //~^ ERROR borrowed value does not live long enough
    //~^^ ERROR `y` does not live long enough
}

fn main() {}
