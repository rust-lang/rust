// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(catch_expr)]

// This test checks that borrows made and returned inside catch blocks are properly constrained
pub fn main() {
    {
        // Test that a borrow which *might* be returned still freezes its referent
        let mut i = 222;
        let x: Result<&i32, ()> = do catch {
            Err(())?;
            &i
        };
        x.ok().cloned();
        i = 0; //~ ERROR cannot assign to `i` because it is borrowed
        let _ = i;
    }

    {
        let x = String::new();
        let _y: Result<(), ()> = do catch {
            Err(())?;
            ::std::mem::drop(x);
        };
        println!("{}", x); //~ ERROR use of moved value: `x`
    }

    {
        // Test that a borrow which *might* be assigned to an outer variable still freezes
        // its referent
        let mut i = 222;
        let j;
        let x: Result<(), ()> = do catch {
            Err(())?;
            j = &i;
        };
        i = 0; //~ ERROR cannot assign to `i` because it is borrowed
        let _ = i;
    }
}

