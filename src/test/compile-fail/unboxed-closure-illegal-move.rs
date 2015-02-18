// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]
#![feature(unboxed_closures)]

// Tests that we can't move out of an unboxed closure environment
// if the upvar is captured by ref or the closure takes self by
// reference.

fn to_fn<A,F:Fn<A>>(f: F) -> F { f }
fn to_fn_mut<A,F:FnMut<A>>(f: F) -> F { f }
fn to_fn_once<A,F:FnOnce<A>>(f: F) -> F { f }

fn main() {
    // By-ref cases
    {
        let x = box 0_usize;
        let f = to_fn(|| drop(x)); //~ ERROR cannot move
    }
    {
        let x = box 0_usize;
        let f = to_fn_mut(|| drop(x)); //~ ERROR cannot move
    }
    {
        let x = box 0_usize;
        let f = to_fn_once(|| drop(x)); // OK -- FnOnce
    }
    // By-value cases
    {
        let x = box 0_usize;
        let f = to_fn(move || drop(x)); //~ ERROR cannot move
    }
    {
        let x = box 0_usize;
        let f = to_fn_mut(move || drop(x)); //~ ERROR cannot move
    }
    {
        let x = box 0_usize;
        let f = to_fn_once(move || drop(x)); // this one is ok
    }
}
