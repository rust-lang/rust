// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we cannot mutate an outer variable that is not declared
// as `mut` through a closure. Also test that we CAN mutate a moved copy,
// unless this is a `Fn` closure. Issue #16749.

#![feature(unboxed_closures)]

use std::mem;

fn to_fn<A,F:Fn<A>>(f: F) -> F { f }
fn to_fn_mut<A,F:FnMut<A>>(f: F) -> F { f }

fn a() {
    let n = 0;
    let mut f = to_fn_mut(|| { //~ ERROR closure cannot assign
        n += 1;
    });
}

fn b() {
    let mut n = 0;
    let mut f = to_fn_mut(|| {
        n += 1; // OK
    });
}

fn c() {
    let n = 0;
    let mut f = to_fn_mut(move || {
        // If we just did a straight-forward desugaring, this would
        // compile, but we do something a bit more subtle, and hence
        // we get an error.
        n += 1; //~ ERROR cannot assign
    });
}

fn d() {
    let mut n = 0;
    let mut f = to_fn_mut(move || {
        n += 1; // OK
    });
}

fn e() {
    let n = 0;
    let mut f = to_fn(move || {
        n += 1; //~ ERROR cannot assign
    });
}

fn f() {
    let mut n = 0;
    let mut f = to_fn(move || {
        n += 1; //~ ERROR cannot assign
    });
}

fn main() { }
