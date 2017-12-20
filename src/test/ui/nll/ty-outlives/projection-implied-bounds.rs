// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Znll -Zborrowck=mir -Zverbose

// Test that we can deduce when projections like `T::Item` outlive the
// function body. Test that this does not imply that `T: 'a` holds.

#![allow(warnings)]
#![feature(rustc_attrs)]

use std::cell::Cell;

fn twice<F, T>(mut value: T, mut f: F)
where
    F: FnMut(&T, Cell<&Option<T::Item>>),
    T: Iterator,
{
    let mut n = value.next();
    f(&value, Cell::new(&n));
    f(&value, Cell::new(&n));
}

#[rustc_errors]
fn generic1<T: Iterator>(value: T) {
    // No error here:
    twice(value, |value_ref, item| invoke1(item));
}

fn invoke1<'a, T>(x: Cell<&'a Option<T>>)
where
    T: 'a,
{
}

#[rustc_errors]
fn generic2<T: Iterator>(value: T) {
    twice(value, |value_ref, item| invoke2(value_ref, item));
    //~^ WARNING not reporting region error due to -Znll
    //~| ERROR the parameter type `T` may not live long enough
}

fn invoke2<'a, T, U>(a: &T, b: Cell<&'a Option<U>>)
where
    T: 'a,
{
}

fn main() {}
