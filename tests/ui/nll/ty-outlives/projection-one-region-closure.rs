// Test cases where we constrain `<T as Anything<'b>>::AssocType` to
// outlive `'a` and there are no bounds in the trait definition of
// `Anything`. This means that the constraint can only be satisfied in two
// ways:
//
// - by ensuring that `T: 'a` and `'b: 'a`, or
// - by something in the where clauses.
//
// As of this writing, the where clause option does not work because
// of limitations in our region inferencing system (this is true both
// with and without NLL). See `projection_outlives`.
//
// Ensuring that both `T: 'a` and `'b: 'a` holds does work (`elements_outlive`).

//@ compile-flags:-Zverbose-internals

#![allow(warnings)]
#![feature(rustc_attrs)]

use std::cell::Cell;

trait Anything<'a> {
    type AssocType;
}

fn with_signature<'a, T, F>(cell: Cell<&'a ()>, t: T, op: F)
where
    F: FnOnce(Cell<&'a ()>, T),
{
    op(cell, t)
}

fn require<'a, 'b, T>(_cell: Cell<&'a ()>, _t: T)
where
    T: Anything<'b>,
    T::AssocType: 'a,
{
}

#[rustc_regions]
fn no_relationships_late<'a, 'b, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b>,
{
    with_signature(cell, t, |cell, t| require(cell, t));
    //~^ ERROR the parameter type `T` may not live long enough
    //~| ERROR
}

#[rustc_regions]
fn no_relationships_early<'a, 'b, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b>,
    'a: 'a,
{
    with_signature(cell, t, |cell, t| require(cell, t));
    //~^ ERROR the parameter type `T` may not live long enough
    //~| ERROR
}

#[rustc_regions]
fn projection_outlives<'a, 'b, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b>,
    T::AssocType: 'a,
{
    // We are projecting `<T as Anything<'b>>::AssocType`, and we know
    // that this outlives `'a` because of the where-clause.

    with_signature(cell, t, |cell, t| require(cell, t));
}

#[rustc_regions]
fn elements_outlive<'a, 'b, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b>,
    T: 'a,
    'b: 'a,
{
    with_signature(cell, t, |cell, t| require(cell, t));
}

fn main() {}
