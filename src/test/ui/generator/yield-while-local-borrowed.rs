// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z borrowck=compare

#![feature(generators, generator_trait)]

use std::ops::{GeneratorState, Generator};
use std::cell::Cell;

unsafe fn borrow_local_inline() {
    // Not OK to yield with a borrow of a temporary.
    //
    // (This error occurs because the region shows up in the type of
    // `b` and gets extended by region inference.)
    let mut b = move || {
        let a = &mut 3;
        //~^ ERROR borrow may still be in use when generator yields (Ast)
        //~| ERROR borrow may still be in use when generator yields (Mir)
        yield();
        println!("{}", a);
    };
    b.resume();
}

unsafe fn borrow_local_inline_done() {
    // No error here -- `a` is not in scope at the point of `yield`.
    let mut b = move || {
        {
            let a = &mut 3;
        }
        yield();
    };
    b.resume();
}

unsafe fn borrow_local() {
    // Not OK to yield with a borrow of a temporary.
    //
    // (This error occurs because the region shows up in the type of
    // `b` and gets extended by region inference.)
    let mut b = move || {
        let a = 3;
        {
            let b = &a;
            //~^ ERROR borrow may still be in use when generator yields (Ast)
            //~| ERROR borrow may still be in use when generator yields (Mir)
            yield();
            println!("{}", b);
        }
    };
    b.resume();
}

fn main() { }
