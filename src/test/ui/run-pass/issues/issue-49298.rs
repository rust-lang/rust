// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(test)]

extern crate test;

enum Void {}

fn main() {
    let mut x: (Void, usize);
    let mut y = 42;
    x.1 = 13;

    // Make sure `y` stays on the stack.
    test::black_box(&mut y);

    // Check that the write to `x.1` did not overwrite `y`.
    // Note that this doesn't fail with optimizations enabled,
    // because we can't keep `x.1` on the stack, like we can `y`,
    // as we can't borrow partially initialized variables.
    assert_eq!(y.to_string(), "42");

    // Check that `(Void, usize)` has space for the `usize` field.
    assert_eq!(std::mem::size_of::<(Void, usize)>(),
               std::mem::size_of::<usize>());
}
