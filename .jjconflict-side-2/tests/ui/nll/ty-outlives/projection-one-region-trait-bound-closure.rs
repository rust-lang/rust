// Test cases where we constrain `<T as Anything<'b>>::AssocType` to
// outlive `'a` and there is a unique bound in the trait definition of
// `Anything` -- i.e., we know that `AssocType` outlives `'b`. In this
// case, the best way to satisfy the trait bound is to show that `'b:
// 'a`, which can be done in various ways.

//@ compile-flags:-Zverbose-internals

#![allow(warnings)]
#![feature(rustc_attrs)]

use std::cell::Cell;

trait Anything<'a> {
    type AssocType: 'a;
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
    //~^ ERROR
}

#[rustc_regions]
fn no_relationships_early<'a, 'b, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b>,
    'a: 'a,
{
    with_signature(cell, t, |cell, t| require(cell, t));
    //~^ ERROR
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
