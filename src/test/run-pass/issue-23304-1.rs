// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

#[repr(u8)]
#[allow(dead_code)]
enum ValueType {
    DOUBLE              = 0x00,
    INT32               = 0x01,
}

#[repr(u32)]
enum ValueTag {
    INT32                = 0x1FFF0u32 | (ValueType::INT32 as u32),
    X,
}

#[repr(u64)]
enum ValueShiftedTag {
    INT32        = ValueTag::INT32 as u64,
    X,
}

fn main() {
    println!("{}", ValueTag::INT32 as u32);
}
