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

pub fn main() {
    // &mut -> &
    let x: &mut int = &mut 42i;
    let x: &int = x;

    let x: &int = &mut 42i;

    // & -> *const
    let x: &int = &42i;
    let x: *const int = x;

    let x: *const int = &42i;

    // &mut -> *const
    let x: &mut int = &mut 42i;
    let x: *const int = x;

    let x: *const int = &mut 42i;

    // *mut -> *const
    let x: *mut int = &mut 42i;
    let x: *const int = x;
}
