// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const JSVAL_TAG_CLEAR: u32 = 0xFFFFFF80;
const JSVAL_TYPE_INT32: u8 = 0x01;
const JSVAL_TYPE_UNDEFINED: u8 = 0x02;
#[repr(u32)]
enum ValueTag {
    JSVAL_TAG_INT32 = JSVAL_TAG_CLEAR | (JSVAL_TYPE_INT32 as u32),
    JSVAL_TAG_UNDEFINED = JSVAL_TAG_CLEAR | (JSVAL_TYPE_UNDEFINED as u32),
}

fn main() {
    let _ = ValueTag::JSVAL_TAG_INT32;
}
