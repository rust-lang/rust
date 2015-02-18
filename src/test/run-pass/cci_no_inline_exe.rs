// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:cci_no_inline_lib.rs

extern crate cci_no_inline_lib;
use cci_no_inline_lib::iter;

pub fn main() {
    // Check that a cross-crate call function not marked as inline
    // does not, in fact, get inlined.  Also, perhaps more
    // importantly, checks that our scheme of using
    // sys::frame_address() to determine if we are inlining is
    // actually working.
    //let bt0 = sys::frame_address();
    //println!("%?", bt0);
    iter(vec!(1_usize, 2_usize, 3_usize), |i| {
        println!("{}", i);

        //let bt1 = sys::frame_address();
        //println!("%?", bt1);

        //assert!(bt0 != bt1);
    })
}
