// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_fn, const_let)]

fn main() {}

const unsafe fn foo(u: *mut u32) {
    std::ptr::write_bytes(u, 0u8, 1);
    //~^ ERROR The use of std::ptr::write_bytes() is gated in constant functions (see issue #53491)
}
