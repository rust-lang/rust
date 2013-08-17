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
// aux-build:cci_iter_lib.rs

extern mod cci_iter_lib;

pub fn main() {
    //let bt0 = sys::rusti::frame_address(1u32);
    //info!("%?", bt0);
    do cci_iter_lib::iter([1, 2, 3]) |i| {
        printf!("%d", *i);
        //assert!(bt0 == sys::rusti::frame_address(2u32));
    }
}
