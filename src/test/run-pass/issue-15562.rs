// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-15562.rs

extern crate i = "issue-15562";

pub fn main() {
    extern {
        fn transmute();
    }
    unsafe {
        transmute();
        i::transmute();
    }
}

// We declare this so we don't run into unresolved symbol errors
// The above extern is NOT `extern "rust-intrinsic"` and thus
// means it'll try to find a corresponding symbol to link to.
#[no_mangle]
pub extern fn transmute() {}
