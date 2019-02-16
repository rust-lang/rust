// Test cases where we constrain `<T as Anything<'a, 'b>>::AssocType`
// to outlive `'a` and there are two bounds in the trait definition of
// `Anything` -- i.e., we know that `AssocType` outlives `'a` and
// `'b`. In this case, it's not clear what is the best way to satisfy
// the trait bound, and hence we propagate it to the caller as a type
// test.

// compile-flags:-Zborrowck=mir -Zverbose

#![allow(warnings)]
#![feature(rustc_attrs)]

use std::cell::Cell;

trait Anything<'a, 'b> {
    type AssocType: 'a + 'b;
}

fn with_signature<'a, T, F>(cell: Cell<&'a ()>, t: T, op: F)
where
    F: FnOnce(Cell<&'a ()>, T),
{
    op(cell, t)
}

fn require<'a, 'b, 'c, T>(_cell: Cell<&'a ()>, _t: T)
where
    T: Anything<'b, 'c>,
    T::AssocType: 'a,
{
}

#[rustc_regions]
fn no_relationships_late<'a, 'b, 'c, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b, 'c>,
{
    with_signature(cell, t, |cell, t| require(cell, t));
    //~^ ERROR associated type `<T as Anything<'_#5r, '_#6r>>::AssocType` may not live long enough
}

#[rustc_regions]
fn no_relationships_early<'a, 'b, 'c, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b, 'c>,
    'a: 'a,
{
    with_signature(cell, t, |cell, t| require(cell, t));
    //~^ ERROR associated type `<T as Anything<'_#6r, '_#7r>>::AssocType` may not live long enough
}

#[rustc_regions]
fn projection_outlives<'a, 'b, 'c, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b, 'c>,
    T::AssocType: 'a,
{
    // We are projecting `<T as Anything<'b>>::AssocType`, and we know
    // that this outlives `'a` because of the where-clause.

    with_signature(cell, t, |cell, t| require(cell, t));
}

#[rustc_regions]
fn elements_outlive1<'a, 'b, 'c, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b, 'c>,
    'b: 'a,
{
    with_signature(cell, t, |cell, t| require(cell, t));
}

#[rustc_regions]
fn elements_outlive2<'a, 'b, 'c, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b, 'c>,
    'c: 'a,
{
    with_signature(cell, t, |cell, t| require(cell, t));
}

#[rustc_regions]
fn two_regions<'a, 'b, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b, 'b>,
{
    with_signature(cell, t, |cell, t| require(cell, t));
    //~^ ERROR lifetime may not live long enough
}

#[rustc_regions]
fn two_regions_outlive<'a, 'b, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b, 'b>,
    'b: 'a,
{
    with_signature(cell, t, |cell, t| require(cell, t));
}

#[rustc_regions]
fn one_region<'a, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'a, 'a>,
{
    // Note that in this case the closure still propagates an external
    // requirement between two variables in its signature, but the
    // creator maps both those two region variables to `'a` on its
    // side.
    with_signature(cell, t, |cell, t| require(cell, t));
}

fn main() {}
