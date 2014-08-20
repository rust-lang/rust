// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]

use std::mem;
use std::ops::Fn;

// Unsugared closure
struct Closure(u8);

impl Fn<(u8,), u8> for Closure {
    extern "rust-call" fn call(&self, (y,): (u8,)) -> u8 {
        let &Closure(x) = self;

        x + y
    }
}

fn main() {
    let y = 0u8;
    let closure = |&: x: u8| x + y;
    let unsugared_closure = Closure(y);

    // Check that both closures are capturing by value
    println!("{}", mem::size_of_val(&closure));  // prints 1
    println!("{}", mem::size_of_val(&unsugared_closure));  // prints 1

    spawn(proc() {
        let ok = unsugared_closure;
        let err = closure;
    })
}
