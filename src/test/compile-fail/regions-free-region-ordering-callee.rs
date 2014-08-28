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

fn ordering1<'a, 'b>(x: &'a &'b uint) -> &'a uint {
    // It is safe to assume that 'a <= 'b due to the type of x
    let y: &'b uint = &**x;
    return y;
}

fn ordering2<'a, 'b>(x: &'a &'b uint, y: &'a uint) -> &'b uint {
    // However, it is not safe to assume that 'b <= 'a
    &*y //~ ERROR cannot infer
}

fn ordering3<'a, 'b>(x: &'a uint, y: &'b uint) -> &'a &'b uint {
    // Do not infer an ordering from the return value.
    let z: &'b uint = &*x;
    //~^ ERROR cannot infer
    fail!();
}

fn ordering4<'a, 'b>(a: &'a uint, b: &'b uint, x: |&'a &'b uint|) {
    // Do not infer ordering from closure argument types.
    let z: Option<&'a &'b uint> = None;
    //~^ ERROR reference has a longer lifetime than the data it references
}

fn ordering5<'a, 'b>(a: &'a uint, b: &'b uint, x: Option<&'a &'b uint>) {
    let z: Option<&'a &'b uint> = None;
}

fn main() {}
