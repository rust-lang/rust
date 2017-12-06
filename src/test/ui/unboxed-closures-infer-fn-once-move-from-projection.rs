// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused)]

fn foo<F>(f: F)
    where F: Fn()
{
}

fn main() {
    // Test that this closure is inferred to `FnOnce` because it moves
    // from `y.0`. This affects the error output (the error is that
    // the closure implements `FnOnce`, not that it moves from inside
    // a `Fn` closure.)
    let y = (vec![1, 2, 3], 0);
    let c = || drop(y.0); //~ ERROR expected a closure that implements the `Fn` trait
    foo(c);
}
