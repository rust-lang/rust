// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators, generator_trait)]

use std::ops::{GeneratorState, Generator};
use std::cell::Cell;

unsafe fn reborrow_shared_ref(x: &i32) {
    // This is OK -- we have a borrow live over the yield, but it's of
    // data that outlives the generator.
    let mut b = move || {
        let a = &*x;
        yield();
        println!("{}", a);
    };
    b.resume();
}

unsafe fn reborrow_mutable_ref(x: &mut i32) {
    // This is OK -- we have a borrow live over the yield, but it's of
    // data that outlives the generator.
    let mut b = move || {
        let a = &mut *x;
        yield();
        println!("{}", a);
    };
    b.resume();
}

unsafe fn reborrow_mutable_ref_2(x: &mut i32) {
    // ...but not OK to go on using `x`.
    let mut b = || {
        let a = &mut *x;
        yield();
        println!("{}", a);
    };
    println!("{}", x); //~ ERROR
    b.resume();
}

fn main() { }
