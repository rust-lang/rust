// compile-flags:-Zborrowck=mir -Zverbose

#![allow(warnings)]
#![feature(rustc_attrs)]

use std::cell::Cell;

// Invoke in such a way that the callee knows:
//
// - 'a: 'x
//
// and it must prove that `T: 'x`. Callee passes along `T: 'a`.
fn twice<'a, F, T>(v: Cell<&'a ()>, value: T, mut f: F)
where
    F: for<'x> FnMut(Option<Cell<&'a &'x ()>>, &T),
{
    f(None, &value);
    f(None, &value);
}

#[rustc_regions]
fn generic<T>(value: T) {
    let cell = Cell::new(&());
    twice(cell, value, |a, b| invoke(a, b));
}

#[rustc_regions]
fn generic_fail<'a, T>(cell: Cell<&'a ()>, value: T) {
    twice(cell, value, |a, b| invoke(a, b));
    //~^ ERROR the parameter type `T` may not live long enough
}

fn invoke<'a, 'x, T>(x: Option<Cell<&'x &'a ()>>, y: &T)
where
    T: 'x,
{
}

fn main() {}
