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
        // Test that borrows returned from a catch block must be valid for the lifetime of the
        // result variable
        let _result: Result<(), &str> = do catch {
            let my_string = String::from("");
            let my_str: & str = & my_string;
            //~^ ERROR `my_string` does not live long enough
            Err(my_str) ?;
            Err("") ?;
        };
    }

    {
        // Test that borrows returned from catch blocks freeze their referent
        let mut i = 5;
        let k = &mut i;
        let mut j: Result<(), &mut i32> = do catch {
            Err(k) ?;
            i = 10; //~ ERROR cannot assign to `i` because it is borrowed
        };
        ::std::mem::drop(k); //~ ERROR use of moved value: `k`
        i = 40; //~ ERROR cannot assign to `i` because it is borrowed

        let i_ptr = if let Err(i_ptr) = j { i_ptr } else { panic ! ("") };
        *i_ptr = 50;
    }
}

