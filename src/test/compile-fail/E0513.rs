// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

fn main() {
    unsafe {
        let size = mem::size_of::<u32>();
        mem::transmute_copy::<u32, [u8; size]>(&8_8); //~ ERROR E0513
                                                      //~| NOTE no type for variable
    }
}
