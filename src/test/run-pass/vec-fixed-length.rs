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

pub fn main() {
    let x: [int, ..4] = [1, 2, 3, 4];
    assert_eq!(x[0], 1);
    assert_eq!(x[1], 2);
    assert_eq!(x[2], 3);
    assert_eq!(x[3], 4);

    assert_eq!(size_of::<[u8, ..4]>(), 4u);

    // FIXME #10183
    if cfg!(target_word_size = "64") {
        assert_eq!(size_of::<[u8, ..(1 << 32)]>(), (1u << 32));
    }
}
