// compile-flags:-Zborrowck=mir -Zverbose
// build-pass (FIXME(62277): could be check-pass?)

// Test that we assume that universal types like `T` outlive the
// function body.

#![allow(warnings)]
#![feature(rustc_attrs)]

use std::cell::Cell;

fn twice<F, T>(value: T, mut f: F)
where
    F: FnMut(Cell<&T>),
{
    f(Cell::new(&value));
    f(Cell::new(&value));
}

#[rustc_errors]
fn generic<T>(value: T) {
    // No error here:
    twice(value, |r| invoke(r));
}

fn invoke<'a, T>(x: Cell<&'a T>)
where
    T: 'a,
{
}

fn main() {}
