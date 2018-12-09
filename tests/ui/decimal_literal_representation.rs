// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[warn(clippy::decimal_literal_representation)]
#[allow(unused_variables)]
fn main() {
    let good = (
        // Hex:
        127,           // 0x7F
        256,           // 0x100
        511,           // 0x1FF
        2048,          // 0x800
        4090,          // 0xFFA
        16_371,        // 0x3FF3
        61_683,        // 0xF0F3
        2_131_750_925, // 0x7F0F_F00D
    );
    let bad = (
        // Hex:
        32_773,        // 0x8005
        65_280,        // 0xFF00
        2_131_750_927, // 0x7F0F_F00F
        2_147_483_647, // 0x7FFF_FFFF
        4_042_322_160, // 0xF0F0_F0F0
    );
}
