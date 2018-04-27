// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
    //~^ WARNING not reporting region error
    //
    // This error from the old region solver looks bogus.
}

#[rustc_regions]
fn generic_fail<'a, T>(cell: Cell<&'a ()>, value: T) {
    twice(cell, value, |a, b| invoke(a, b));
    //~^ WARNING not reporting region error
    //~| WARNING not reporting region error
    //~| ERROR the parameter type `T` may not live long enough
}

fn invoke<'a, 'x, T>(x: Option<Cell<&'x &'a ()>>, y: &T)
where
    T: 'x,
{
}

fn main() {}
