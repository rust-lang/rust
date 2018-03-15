// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::mem::size_of;

// The two enums that follow are designed so that bugs trigger layout optimization.
// Specifically, if either of the following reprs used here is not detected by the compiler,
// then the sizes will be wrong.

#[repr(C, u8)]
enum E1 {
    A(u8, u16, u8),
    B(u8, u16, u8)
}

#[repr(u8, C)]
enum E2 {
    A(u8, u16, u8),
    B(u8, u16, u8)
}

// From pr 37429

#[repr(C,packed)]
pub struct p0f_api_query {
    pub magic: u32,
    pub addr_type: u8,
    pub addr: [u8; 16],
}

pub fn main() {
    assert_eq!(size_of::<E1>(), 6);
    assert_eq!(size_of::<E2>(), 6);
    assert_eq!(size_of::<p0f_api_query>(), 21);
}
