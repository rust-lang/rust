// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs, core_intrinsics)]

#[rustc_mir]
fn zeroed() -> [u64; 5] {
    unsafe {
    ::std::intrinsics::init()
    }
}

#[rustc_mir]
fn bswappd() -> u16 {
    unsafe {
    ::std::intrinsics::bswap(0xABCD)
    }
}

fn main() {
  assert_eq!(zeroed(), [0, 0, 0, 0, 0]);
  assert_eq!(bswappd(), 0xCDAB);
}
