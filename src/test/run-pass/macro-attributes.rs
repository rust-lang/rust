// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty - token trees can't pretty print

#![feature(macro_rules)]

macro_rules! compiles_fine {
    (#[$at:meta]) => {
        // test that the different types of attributes work
        #[attribute]
        /// Documentation!
        #[$at]

        // check that the attributes are recognised by requiring this
        // to be removed to avoid a compile error
        #[cfg(always_remove)]
        static MISTYPED: () = "foo";
    }
}

// item
compiles_fine!(#[foo])

pub fn main() {
    // statement
    compiles_fine!(#[bar]);
}
