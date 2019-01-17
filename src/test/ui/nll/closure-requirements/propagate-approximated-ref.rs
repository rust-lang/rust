// Rather convoluted setup where we infer a relationship between two
// free regions in the closure signature (`'a` and `'b`) on the basis
// of a relationship between two bound regions (`'x` and `'y`).
//
// The idea is that, thanks to invoking `demand_y`, `'x: 'y` must
// hold, where `'x` and `'y` are bound regions. The closure can't
// prove that directly, and because `'x` and `'y` are bound it cannot
// ask the caller to prove it either. But it has bounds on `'x` and
// `'y` in terms of `'a` and `'b`, and it can propagate a relationship
// between `'a` and `'b` to the caller.
//
// Note: the use of `Cell` here is to introduce invariance. One less
// variable.

// compile-flags:-Zborrowck=mir -Zverbose

#![feature(rustc_attrs)]

use std::cell::Cell;

// Callee knows that:
//
// 'x: 'a
// 'b: 'y
//
// so if we are going to ensure that `'x: 'y`, then `'a: 'b` must
// hold.
fn establish_relationships<'a, 'b, F>(_cell_a: &Cell<&'a u32>, _cell_b: &Cell<&'b u32>, _closure: F)
where
    F: for<'x, 'y> FnMut(
        &Cell<&'a &'x u32>, // shows that 'x: 'a
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
        //~^ ERROR lifetime may not live long enough
    });
}

fn main() {}
