// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const U8_MAX_HALF: u8 = !0u8 / 2;
const U16_MAX_HALF: u16 = !0u16 / 2;
const U32_MAX_HALF: u32 = !0u32 / 2;
const U64_MAX_HALF: u64 = !0u64 / 2;

fn main() {
    assert_eq!(U8_MAX_HALF, 0x7f);
    assert_eq!(U16_MAX_HALF, 0x7fff);
    assert_eq!(U32_MAX_HALF, 0x7fff_ffff);
    assert_eq!(U64_MAX_HALF, 0x7fff_ffff_ffff_ffff);
}
