// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_variables)]

use std::ptr;

// Test that we only report **one** error here and that is that
// `break` with an expression is illegal in this context. In
// particular, we don't report any mismatched types error, which is
// besides the point.

fn main() {
    for _ in &[1,2,3] {
        break 22 //~ ERROR `break` with value from a `for` loop
    }
}
