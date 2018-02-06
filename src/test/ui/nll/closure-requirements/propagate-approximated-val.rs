// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A simpler variant of `outlives-from-argument` where cells are
// passed by value.
//
// This is simpler because there are no "extraneous" region
// relationships. In the 'main' variant, there are a number of
// anonymous regions as well.

// compile-flags:-Znll -Zborrowck=mir -Zverbose

#![feature(rustc_attrs)]

use std::cell::Cell;

// Callee knows that:
//
// 'x: 'a
// 'b: 'y
//
// so if we are going to ensure that `'x: 'y`, then `'a: 'b` must
// hold.
fn establish_relationships<'a, 'b, F>(_cell_a: Cell<&'a u32>, _cell_b: Cell<&'b u32>, _closure: F)
where
    F: for<'x, 'y> FnMut(
        Cell<&'a &'x u32>, // shows that 'x: 'a
        Cell<&'y &'b u32>, // shows that 'b: 'y
        Cell<&'x u32>,
        Cell<&'y u32>,
    ),
{
}

fn demand_y<'x, 'y>(_outlives1: Cell<&&'x u32>, _outlives2: Cell<&'y &u32>, _y: &'y u32) {}

#[rustc_regions]
fn test<'a, 'b>(cell_a: Cell<&'a u32>, cell_b: Cell<&'b u32>) {
    establish_relationships(cell_a, cell_b, |outlives1, outlives2, x, y| {
        //~^ ERROR lifetime mismatch

        // Only works if 'x: 'y:
        demand_y(outlives1, outlives2, x.get()) //~ WARNING not reporting region error due to -Znll
    });
}

fn main() {}
