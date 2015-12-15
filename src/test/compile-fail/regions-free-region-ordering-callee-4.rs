// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that callees correctly infer an ordering between free regions
// that appear in their parameter list.  See also
// regions-free-region-ordering-caller.rs

fn ordering4<'a, 'b, F>(a: &'a usize, b: &'b usize, x: F) where F: FnOnce(&'a &'b usize) {
    //~^ ERROR reference has a longer lifetime than the data it references
    // Do not infer ordering from closure argument types.
    let z: Option<&'a &'b usize> = None;
}

fn main() {}
