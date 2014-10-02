// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test mutability and slicing syntax.

#![feature(slicing_syntax)]

fn main() {
    let x: &[int] = &[1, 2, 3, 4, 5];
    // Immutable slices are not mutable.
    let y: &mut[_] = x[2..4]; //~ ERROR cannot borrow immutable dereference of `&`-pointer as mutabl

    let x: &mut [int] = &mut [1, 2, 3, 4, 5];
    // Can't borrow mutably twice
    let y = x[mut 1..2];
    let y = x[mut 4..5]; //~ERROR cannot borrow
}
