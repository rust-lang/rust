// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast - check-fast doesn't understand aux-build
// aux-build:cci_impl_lib.rs

extern mod cci_impl_lib;
use cci_impl_lib::uint_helpers;

pub fn main() {
    //let bt0 = sys::frame_address();
    //info!("%?", bt0);

    do 3u.to(10u) |i| {
        printfln!("%u", i);

        //let bt1 = sys::frame_address();
        //info!("%?", bt1);
        //assert!(bt0 == bt1);
    }
}
