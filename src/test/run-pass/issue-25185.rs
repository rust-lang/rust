// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-25185-1.rs
// aux-build:issue-25185-2.rs
// ignore-wasm32-bare no libc for ffi testing

extern crate issue_25185_2;

fn main() {
    let x = unsafe {
        issue_25185_2::rust_dbg_extern_identity_u32(1)
    };
    assert_eq!(x, 1);
}
