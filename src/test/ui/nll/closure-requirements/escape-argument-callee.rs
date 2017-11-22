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
// *but* the signature of the closure doesn't indicate that `y` lives
// long enough for that. The closure reports the error (and hence we
// see it before the closure's "external requirements" report).

// compile-flags:-Znll -Zborrowck=mir -Zverbose

#![feature(rustc_attrs)]

#[rustc_regions]
fn test() {
    let x = 44;
    let mut p = &x;

    {
        let y = 22;
        let mut closure = expect_sig(|p, y| *p = y);
        //~^ ERROR free region `'_#4r` does not outlive free region `'_#3r`
        //~| WARNING not reporting region error due to -Znll
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
