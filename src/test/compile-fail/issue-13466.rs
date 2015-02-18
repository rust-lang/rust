// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #13466

pub fn main() {
    // The expected arm type `Option<T>` has one type parameter, while
    // the actual arm `Result<T, E>` has two. typeck should not be
    // tricked into looking up a non-existing second type parameter.
    let _x: usize = match Some(1_usize) {
        Ok(u) => u,
        //~^ ERROR mismatched types
        //~| expected `core::option::Option<usize>`
        //~| found `core::result::Result<_, _>`
        //~| expected enum `core::option::Option`
        //~| found enum `core::result::Result`

        Err(e) => panic!(e)
        //~^ ERROR mismatched types
        //~| expected `core::option::Option<usize>`
        //~| found `core::result::Result<_, _>`
        //~| expected enum `core::option::Option`
        //~| found enum `core::result::Result`
    };
}
