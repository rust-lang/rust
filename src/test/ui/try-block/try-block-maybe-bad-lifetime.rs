// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --edition 2018

#![feature(try_blocks)]

#![inline(never)]
fn do_something_with<T>(_x: T) {}

// This test checks that borrows made and returned inside try blocks are properly constrained
pub fn main() {
    {
        // Test that a borrow which *might* be returned still freezes its referent
        let mut i = 222;
        let x: Result<&i32, ()> = try {
            Err(())?;
            &i
        };
        i = 0; //~ ERROR cannot assign to `i` because it is borrowed
        let _ = i;
        do_something_with(x);
    }

    {
        let x = String::new();
        let _y: Result<(), ()> = try {
            Err(())?;
            ::std::mem::drop(x);
        };
        println!("{}", x); //~ ERROR borrow of moved value: `x`
    }

    {
        // Test that a borrow which *might* be assigned to an outer variable still freezes
        // its referent
        let mut i = 222;
        let mut j = &-1;
        let _x: Result<(), ()> = try {
            Err(())?;
            j = &i;
        };
        i = 0; //~ ERROR cannot assign to `i` because it is borrowed
        let _ = i;
        do_something_with(j);
    }
}

