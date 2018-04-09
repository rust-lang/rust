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
// - takes an argument `y` with lifetime `'a` (in the code, it's anonymous)
// - stores `y` into another, longer-lived spot with lifetime `'b`
//
// Because `'a` and `'b` are two different, unrelated higher-ranked
// regions with no relationship to one another, this is an error. This
// error is reported by the closure itself and is not propagated to
// its creator: this is because `'a` and `'b` are higher-ranked
// (late-bound) regions and the closure is not allowed to propagate
// additional where clauses between higher-ranked regions, only those
// that appear free in its type (hence, we see it before the closure's
// "external requirements" report).

// compile-flags:-Zborrowck=mir -Zverbose

#![feature(rustc_attrs)]

#[rustc_regions]
fn test() {
    let x = 44;
    let mut p = &x;

    {
        let y = 22;
        let mut closure = expect_sig(|p, y| *p = y);
        //~^ ERROR does not outlive free region
        //~| WARNING not reporting region error due to nll
        closure(&mut p, &y);
    }

    deref(p);
}

fn expect_sig<F>(f: F) -> F
    where F: FnMut(&mut &i32, &i32)
{
    f
}

fn deref(_p: &i32) { }

fn main() { }
