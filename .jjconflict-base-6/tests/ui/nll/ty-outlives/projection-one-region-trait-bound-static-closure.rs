// Test cases where we constrain `<T as Anything<'b>>::AssocType` to
// outlive `'static`. In this case, we don't get any errors, and in fact
// we don't even propagate constraints from the closures to the callers.

//@ compile-flags:-Zverbose-internals
//@ check-pass

#![allow(warnings)]
#![feature(rustc_attrs)]

use std::cell::Cell;

trait Anything<'a> {
    type AssocType: 'static;
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
}

#[rustc_regions]
fn no_relationships_early<'a, 'b, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b>,
    'a: 'a,
{
    with_signature(cell, t, |cell, t| require(cell, t));
}

#[rustc_regions]
fn projection_outlives<'a, 'b, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b>,
    T::AssocType: 'a,
{
    // This error is unfortunate. This code ought to type-check: we
    // are projecting `<T as Anything<'b>>::AssocType`, and we know
    // that this outlives `'a` because of the where-clause. However,
    // the way the region checker works, we don't register this
    // outlives obligation, and hence we get an error: this is because
    // what we see is a projection like `<T as
    // Anything<'?0>>::AssocType`, and we don't yet know if `?0` will
    // equal `'b` or not, so we ignore the where-clause. Obviously we
    // can do better here with a more involved verification step.

    with_signature(cell, t, |cell, t| require(cell, t));
}

#[rustc_regions]
fn elements_outlive<'a, 'b, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b>,
    'b: 'a,
{
    with_signature(cell, t, |cell, t| require(cell, t));
}

#[rustc_regions]
fn one_region<'a, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'a>,
{
    // Note that in this case the closure still propagates an external
    // requirement between two variables in its signature, but the
    // creator maps both those two region variables to `'a` on its
    // side.
    with_signature(cell, t, |cell, t| require(cell, t));
}

fn main() {}
