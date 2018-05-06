// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test cases where we constrain `<T as Anything<'b>>::AssocType` to
// outlive `'a` and there is a unique bound in the trait definition of
// `Anything` -- i.e., we know that `AssocType` outlives `'b`. In this
// case, the best way to satisfy the trait bound is to show that `'b:
// 'a`, which can be done in various ways.

// compile-flags:-Zborrowck=mir -Zverbose

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
    //~^ WARNING not reporting region error due to nll
    //~| ERROR does not outlive free region
}

#[rustc_regions]
fn no_relationships_early<'a, 'b, T>(cell: Cell<&'a ()>, t: T)
where
    T: Anything<'b>,
    'a: 'a,
{
    with_signature(cell, t, |cell, t| require(cell, t));
    //~^ WARNING not reporting region error due to nll
    //~| ERROR does not outlive free region
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
    //~^ WARNING not reporting region error due to nll
    //~| ERROR does not outlive free region
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
