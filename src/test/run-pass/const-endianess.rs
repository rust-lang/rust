// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_int_ops)]
#![feature(test)]

extern crate test;
use test::black_box as b;

const BE_U32: u32 = 55u32.to_be();
const LE_U32: u32 = 55u32.to_le();


fn main() {
    assert_eq!(BE_U32, b(55u32).to_be());
    assert_eq!(LE_U32, b(55u32).to_le());

    #[cfg(not(target_arch = "asmjs"))]
    {
        const BE_U128: u128 = 999999u128.to_be();
        const LE_I128: i128 = -999999i128.to_le();
        assert_eq!(BE_U128, b(999999u128).to_be());
        assert_eq!(LE_I128, b(-999999i128).to_le());
    }
}
