// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that placement in respects unstable code checks.

#![feature(placement_in_syntax)]
#![feature(core)]

extern crate core;

fn main() {
    use std::boxed::HEAP; //~ ERROR use of unstable library feature

    let _ = in HEAP { //~ ERROR use of unstable library feature
        ::core::raw::Slice { //~ ERROR use of unstable library feature
            data: &42, //~ ERROR use of unstable library feature
            len: 1 //~ ERROR use of unstable library feature
        }
    };
}
