// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name="foo"]

use std::mem::size_of;

// compile-flags: -Z fuel=foo=1

struct S1(u8, u16, u8);
struct S2(u8, u16, u8);

fn main() {
    let optimized = (size_of::<S1>() == 4) as usize
        +(size_of::<S2>() == 4) as usize;
    assert_eq!(optimized, 1);
}
