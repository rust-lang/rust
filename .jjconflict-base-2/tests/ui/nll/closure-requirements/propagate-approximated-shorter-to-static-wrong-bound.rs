// Test a case where we are trying to prove `'x: 'y` and are forced to
// approximate the shorter end-point (`'y`) to with `'static`. This is
// because `'y` is higher-ranked but we know of only irrelevant
// relations to other regions. Note that `'static` shows up in the
// stderr output as `'0`.

//@ compile-flags:-Zverbose-internals

#![feature(rustc_attrs)]

use std::cell::Cell;

// Callee knows that:
//
// 'x: 'a
// 'y: 'b
//
// so the only way we can ensure that `'x: 'y` is to show that
// `'a: 'static`.
fn establish_relationships<'a, 'b, F>(_cell_a: &Cell<&'a u32>, _cell_b: &Cell<&'b u32>, _closure: F)
where
    F: for<'x, 'y> FnMut(
        &Cell<&'a &'x u32>, // shows that 'x: 'a
        &Cell<&'b &'y u32>, // shows that 'y: 'b
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
        //~^ ERROR borrowed data escapes outside of function
    });
}

fn main() {}
