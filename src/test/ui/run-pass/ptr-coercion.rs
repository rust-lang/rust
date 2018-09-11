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

// pretty-expanded FIXME #23616

pub fn main() {
    // &mut -> &
    let x: &mut isize = &mut 42;
    let x: &isize = x;

    let x: &isize = &mut 42;

    // & -> *const
    let x: &isize = &42;
    let x: *const isize = x;

    let x: *const isize = &42;

    // &mut -> *const
    let x: &mut isize = &mut 42;
    let x: *const isize = x;

    let x: *const isize = &mut 42;

    // *mut -> *const
    let x: *mut isize = &mut 42;
    let x: *const isize = x;
}
