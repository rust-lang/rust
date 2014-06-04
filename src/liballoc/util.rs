// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![doc(hidden)]

use core::mem;
use core::raw;

#[inline]
#[deprecated]
pub fn get_box_size(body_size: uint, body_align: uint) -> uint {
    let header_size = mem::size_of::<raw::Box<()>>();
    let total_size = align_to(header_size, body_align) + body_size;
    total_size
}

// Rounds size to the next alignment. Alignment is required to be a power of
// two.
#[inline]
fn align_to(size: uint, align: uint) -> uint {
    assert!(align != 0);
    (size + align - 1) & !(align - 1)
}
