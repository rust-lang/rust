// Test where we fail to approximate due to demanding a postdom
// relationship between our upper bounds.

// compile-flags:-Zborrowck=mir -Zverbose

#![feature(rustc_attrs)]

use std::cell::Cell;

// Callee knows that:
//
// 'x: 'a
// 'x: 'b
// 'c: 'y
//
// we have to prove that `'x: 'y`. We currently can only approximate
// via a postdominator -- hence we fail to choose between `'a` and
// `'b` here and report the error in the closure.
fn establish_relationships<'a, 'b, 'c, F>(
    _cell_a: Cell<&'a u32>,
    _cell_b: Cell<&'b u32>,
    _cell_c: Cell<&'c u32>,
    _closure: F,
) where
    F: for<'x, 'y> FnMut(
        Cell<&'a &'x u32>, // shows that 'x: 'a
        Cell<&'b &'x u32>, // shows that 'x: 'b
        Cell<&'y &'c u32>, // shows that 'c: 'y
        Cell<&'x u32>,
        Cell<&'y u32>,
    ),
{
}

fn demand_y<'x, 'y>(_cell_x: Cell<&'x u32>, _cell_y: Cell<&'y u32>, _y: &'y u32) {}

#[rustc_regions]
fn supply<'a, 'b, 'c>(cell_a: Cell<&'a u32>, cell_b: Cell<&'b u32>, cell_c: Cell<&'c u32>) {
    establish_relationships(
        cell_a,
        cell_b,
        cell_c,
        |_outlives1, _outlives2, _outlives3, x, y| {
            // Only works if 'x: 'y:
            let p = x.get();
            demand_y(x, y, p) //~ ERROR
        },
    );
}

fn main() {}
