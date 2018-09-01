// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_int_rotate)]

const LEFT: u32 = 0x10000b3u32.rotate_left(8);
const RIGHT: u32 = 0xb301u32.rotate_right(8);

fn main() {
    assert_eq!(LEFT, 0xb301);
    assert_eq!(RIGHT, 0x10000b3);
}
