// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//error-pattern: tried to access memory with alignment 1, but alignment 2 is required

fn main() {
    let mut data = [0u16; 8];
    let ptr = (&mut data[0] as *mut u16 as *mut u8).wrapping_add(1) as *mut u16;
    // Even copying 0 elements to something unaligned should error
    unsafe { ptr.copy_from(&data[5], 0); }
}
