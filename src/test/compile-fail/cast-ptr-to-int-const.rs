// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-const_raw_ptr_to_usize_cast

fn main() {
    const X: u32 = main as u32; //~ ERROR casting pointers to integers in constants is unstable
    const Y: u32 = 0;
    const Z: u32 = &Y as *const u32 as u32; //~ ERROR is unstable
}
