// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we assume that universal types like `T` outlive the
// function body. Same as ty-param-fn-body, but uses `feature(nll)`,
// which affects error reporting.

#![feature(nll)]

#![allow(warnings)]

use std::cell::Cell;

// No errors here, because `'a` is local to the body.
fn region_within_body<T>(t: T) {
    let some_int = 22;
    let cell = Cell::new(&some_int);
    outlives(cell, t)
}

// Error here, because T: 'a is not satisfied.
fn region_static<'a, T>(cell: Cell<&'a usize>, t: T) {
    outlives(cell, t)
    //~^ ERROR the parameter type `T` may not live long enough
}

fn outlives<'a, T>(x: Cell<&'a usize>, y: T)
where
    T: 'a,
{
}

fn main() {}
