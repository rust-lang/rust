// Test a case where we setup relationships like `'x: 'a` or `'a: 'x`,
// where `'x` is bound in closure type but `'a` is free. This forces
// us to approximate `'x` one way or the other.

//@ compile-flags:-Zverbose-internals

#![feature(rustc_attrs)]

use std::cell::Cell;

fn foo<'a, F>(_cell: Cell<&'a u32>, _f: F)
where
    F: for<'x> FnOnce(Cell<&'a u32>, Cell<&'x u32>),
{
}

#[rustc_regions]
fn case1() {
    let a = 0;
    let cell = Cell::new(&a);
    foo(cell, |cell_a, cell_x| {
        cell_a.set(cell_x.get()); // forces 'x: 'a, error in closure
        //~^ ERROR
    })
}

#[rustc_regions]
fn case2() {
    let a = 0;
    let cell = Cell::new(&a);
    //~^ ERROR `a` does not live long enough

    // As you can see in the stderr output, this closure propoagates a
    // requirement that `'a: 'static'.
    foo(cell, |cell_a, cell_x| {
        cell_x.set(cell_a.get()); // forces 'a: 'x, implies 'a = 'static -> borrow error
    })
}

fn main() { }
