// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
    //~^ WARNING not reporting region error due to nll
    //~| ERROR associated type `<T as Anything<'_#5r, '_#6r>>::AssocType` may not live long enough
}

#[rustc_regions]
fn no_relationships_early<'a, 'b, 'c, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b, 'c>,
    'a: 'a,
{
    with_signature(cell, t, |cell, t| require(cell, t));
    //~^ WARNING not reporting region error due to nll
    //~| ERROR associated type `<T as Anything<'_#6r, '_#7r>>::AssocType` may not live long enough
}

#[rustc_regions]
fn projection_outlives<'a, 'b, 'c, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b, 'c>,
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
    //~^ WARNING not reporting region error due to nll
    //~| ERROR associated type `<T as Anything<'_#6r, '_#7r>>::AssocType` may not live long enough
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
    //~^ WARNING not reporting region error due to nll
    //~| ERROR does not outlive free region
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
