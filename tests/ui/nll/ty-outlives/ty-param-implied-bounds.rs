//@ compile-flags:-Zverbose-internals
//@ check-pass

// Test that we assume that universal types like `T` outlive the
// function body.

use std::cell::Cell;

fn twice<F, T>(value: T, mut f: F)
where
    F: FnMut(Cell<&T>),
{
    f(Cell::new(&value));
    f(Cell::new(&value));
}

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
