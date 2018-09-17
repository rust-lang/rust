// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//error-pattern: invalid use of NULL pointer

fn main() {
    let mut data = [0u16; 4];
    let ptr = &mut data[0] as *mut u16;
    // Even copying 0 elements from NULL should error
    unsafe { ptr.copy_from(std::ptr::null(), 0); }
}
