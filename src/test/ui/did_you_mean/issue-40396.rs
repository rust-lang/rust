// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo() {
    (0..13).collect<Vec<i32>>();
    //~^ ERROR chained comparison
    //~| ERROR expected value, found struct `Vec`
    //~| ERROR expected value, found builtin type `i32`
    //~| ERROR attempted to take value of method `collect`
}

fn bar() {
    Vec<i32>::new();
    //~^ ERROR chained comparison
    //~| ERROR expected value, found struct `Vec`
    //~| ERROR expected value, found builtin type `i32`
    //~| ERROR cannot find function `new` in the crate root
}

fn qux() {
    (0..13).collect<Vec<i32>();
    //~^ ERROR chained comparison
    //~| ERROR chained comparison
    //~| ERROR expected value, found struct `Vec`
    //~| ERROR expected value, found builtin type `i32`
    //~| ERROR attempted to take value of method `collect`
    //~| ERROR mismatched types
}

fn main() {}
