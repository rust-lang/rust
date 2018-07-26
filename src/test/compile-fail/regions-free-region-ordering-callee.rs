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

fn ordering1<'a, 'b>(x: &'a &'b usize) -> &'a usize {
    // It is safe to assume that 'a <= 'b due to the type of x
    let y: &'b usize = &**x;
    return y;
}

fn ordering2<'a, 'b>(x: &'a &'b usize, y: &'a usize) -> &'b usize {
    // However, it is not safe to assume that 'b <= 'a
    &*y //~ ERROR 23:5: 23:8: lifetime mismatch [E0623]
}

fn ordering3<'a, 'b>(x: &'a usize, y: &'b usize) -> &'a &'b usize {
    // Do not infer an ordering from the return value.
    let z: &'b usize = &*x;
    //~^ ERROR 28:24: 28:27: lifetime mismatch [E0623]
    panic!();
}

// see regions-free-region-ordering-callee-4.rs

fn ordering5<'a, 'b>(a: &'a usize, b: &'b usize, x: Option<&'a &'b usize>) {
    let z: Option<&'a &'b usize> = None;
}

fn main() {}
