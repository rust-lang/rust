// compile-flags:-Zborrowck=mir

// Test that we assume that universal types like `T` outlive the
// function body.

#![allow(warnings)]

use std::cell::Cell;

// No errors here, because `'a` is local to the body.
fn region_within_body<T>(t: T) {
    let some_int = 22;
    let cell = Cell::new(&some_int);
    outlives(cell, t)
}

// Error here, because T: 'a is not satisfied.
fn region_static<'a, T>(cell: Cell<&'a usize>, t: T) {
    outlives(cell, t)
    //~^ ERROR the parameter type `T` may not live long enough
}

fn outlives<'a, T>(x: Cell<&'a usize>, y: T)
where
    T: 'a,
{
}

fn main() {}
