// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test closure that:
//
// - captures a variable `y` by reference
// - stores that reference to `y` into another, longer-lived place (`p`)
//
// Both of these are upvars of reference type (the capture of `y` is
// of type `&'a i32`, the capture of `p` is of type `&mut &'b
// i32`). The closure thus computes a relationship between `'a` and
// `'b`.  This relationship is propagated to the closure creator,
// which reports an error.

// compile-flags:-Zborrowck=mir -Zverbose

#![feature(rustc_attrs)]

#[rustc_regions]
fn test() {
    let x = 44;
    let mut p = &x;

    {
        let y = 22;
        let mut closure = || p = &y;
        //~^ ERROR `y` does not live long enough [E0597]
        closure();
    }

    deref(p);
}

fn deref(_p: &i32) { }

fn main() { }
