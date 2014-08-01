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

// Test references to items that haven't been translated yet.

// Generate some code in the first compilation unit before declaring any
// modules.  This ensures that the first module doesn't go into the same
// compilation unit as the top-level module.
fn pad() -> uint { 0 }

mod b {
    pub fn three() -> uint {
        ::one() + ::a::two()
    }
}

mod a {
    pub fn two() -> uint {
        ::one() + ::one()
    }
}

fn one() -> uint {
    1
}

fn main() {
    assert_eq!(one(), 1);
    assert_eq!(a::two(), 2);
    assert_eq!(b::three(), 3);
}

