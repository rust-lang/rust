// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Similarly to escape-argument-callee, a test case where the closure
// requires a relationship between 2 unrelated higher-ranked regions,
// with no helpful relations between the HRRs and free regions.
//
// In this case, the error is reported by the closure itself. This is
// because it is unable to approximate the higher-ranked region `'x`,
// as it only knows of regions that `'x` is outlived by, and none that
// `'x` outlives.

// compile-flags:-Znll -Zborrowck=mir -Zverbose

#![feature(rustc_attrs)]

use std::cell::Cell;

// Callee knows that:
//
// 'a: 'x
// 'b: 'y
//
// but this doesn't really help us in proving that `'x: 'y`, so
// closure gets an error.  In particular, we would need to know that
// `'x: 'a`, so that we could approximate `'x` "downwards" to `'a`.
fn establish_relationships<'a, 'b, F>(_cell_a: &Cell<&'a u32>, _cell_b: &Cell<&'b u32>, _closure: F)
where
    F: for<'x, 'y> FnMut(
        &Cell<&'x &'a u32>, // shows that 'a: 'x
        &Cell<&'y &'b u32>, // shows that 'b: 'y
        &Cell<&'x u32>,
        &Cell<&'y u32>,
    ),
{
}

fn demand_y<'x, 'y>(_cell_x: &Cell<&'x u32>, _cell_y: &Cell<&'y u32>, _y: &'y u32) {}

#[rustc_regions]
fn supply<'a, 'b>(cell_a: Cell<&'a u32>, cell_b: Cell<&'b u32>) {
    establish_relationships(&cell_a, &cell_b, |_outlives1, _outlives2, x, y| {
        // Only works if 'x: 'y:
        demand_y(x, y, x.get())
        //~^ WARN not reporting region error due to -Znll
        //~| ERROR does not outlive free region
    });
}

fn main() {}
