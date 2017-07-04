// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(exceeding_bitshifts, const_err)]

fn main() {
    // Make sure we catch overflows that would be hidden by first casting the RHS to u32
    let _n = 1i64 >> (u32::max_value() as i64 + 1); //~ Overflow(Shr)
}
