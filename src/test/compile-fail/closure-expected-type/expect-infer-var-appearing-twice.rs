// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn with_closure<F, A>(_: F)
    where F: FnOnce(A, A)
{
}

fn a() {
    with_closure(|x: u32, y| {
        // We deduce type of `y` from `x`.
    });
}

fn b() {
    // Here we take the supplied types, resulting in an error later on.
    with_closure(|x: u32, y: i32| {
        //~^ ERROR type mismatch in closure arguments
    });
}

fn c() {
    with_closure(|x, y: i32| {
        // We deduce type of `x` from `y`.
    });
}

fn main() { }
