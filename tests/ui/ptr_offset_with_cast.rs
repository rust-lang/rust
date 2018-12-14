// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let vec = vec![b'a', b'b', b'c'];
    let ptr = vec.as_ptr();

    let offset_u8 = 1_u8;
    let offset_usize = 1_usize;
    let offset_isize = 1_isize;

    unsafe {
        ptr.offset(offset_usize as isize);
        ptr.offset(offset_isize as isize);
        ptr.offset(offset_u8 as isize);

        ptr.wrapping_offset(offset_usize as isize);
        ptr.wrapping_offset(offset_isize as isize);
        ptr.wrapping_offset(offset_u8 as isize);
    }
}
