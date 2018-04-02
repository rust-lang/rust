// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Encountered while testing #44614.
// compile-pass

pub fn main() {
    // Constant of generic type (int)
    const X: &'static u32 = &22;
    assert_eq!(0, match &22 {
        X => 0,
        _ => 1,
    });
}
