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
// - takes an argument `y`
// - stores `y` into another, longer-lived spot
//
// but is invoked with a spot that doesn't live long
// enough to store `y`.
//
// The error is reported in the caller: invoking the closure links the
// lifetime of the variable that is given as `y` (via subtyping) and
// thus forces the corresponding borrow to live too long. This is
// basically checking that the MIR type checker correctly enforces the
// closure signature.

// compile-flags:-Zborrowck=mir -Zverbose

#![feature(rustc_attrs)]

#[rustc_regions]
fn test() {
    let x = 44;
    let mut p = &x;

    {
        let y = 22;
        let mut closure = expect_sig(|p, y| *p = y);
        closure(&mut p, &y);
        //~^ ERROR `y` does not live long enough [E0597]
    }

    deref(p);
}

fn expect_sig<F>(f: F) -> F
    where F: for<'a, 'b> FnMut(&'a mut &'b i32, &'b i32)
{
    f
}

fn deref(_p: &i32) { }

fn main() { }
