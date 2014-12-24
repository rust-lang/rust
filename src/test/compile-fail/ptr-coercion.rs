// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test coercions between pointers which don't do anything fancy like unsizing.
// These are testing that we don't lose mutability when converting to raw pointers.

pub fn main() {
    // *const -> *mut
    let x: *const int = &42i;
    let x: *mut int = x; //~ERROR values differ in mutability

    // & -> *mut
    let x: *mut int = &42; //~ERROR values differ in mutability

    let x: *const int = &42;
    let x: *mut int = x; //~ERROR values differ in mutability
}
