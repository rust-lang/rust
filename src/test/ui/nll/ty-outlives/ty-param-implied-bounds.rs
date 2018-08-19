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
// compile-pass

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
