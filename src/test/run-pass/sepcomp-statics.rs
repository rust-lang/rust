// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C codegen-units=3

// Test references to static items across compilation units.

fn pad() -> uint { 0 }

static ONE: uint = 1;

mod b {
    // Separate compilation always switches to the LLVM module with the fewest
    // instructions.  Make sure we have some instructions in this module so
    // that `a` and `b` don't go into the same compilation unit.
    fn pad() -> uint { 0 }

    pub static THREE: uint = ::ONE + ::a::TWO;
}

mod a {
    fn pad() -> uint { 0 }

    pub static TWO: uint = ::ONE + ::ONE;
}

fn main() {
    assert_eq!(ONE, 1);
    assert_eq!(a::TWO, 2);
    assert_eq!(b::THREE, 3);
}

