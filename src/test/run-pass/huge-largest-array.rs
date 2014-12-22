// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem::size_of;

#[cfg(target_word_size = "32")]
pub fn main() {
    assert_eq!(size_of::<[u8, ..(1 << 31) - 1]>(), (1 << 31) - 1);
}

#[cfg(target_word_size = "64")]
pub fn main() {
    assert_eq!(size_of::<[u8, ..(1 << 47) - 1]>(), (1 << 47) - 1);
}
