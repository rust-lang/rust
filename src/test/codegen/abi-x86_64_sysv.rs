// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// only-x86_64

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

pub struct S24 {
  a: i8,
  b: i8,
  c: i8,
}

pub struct S48 {
  a: i16,
  b: i16,
  c: i8,
}

// CHECK: i24 @struct_24_bits(i24
#[no_mangle]
pub extern "sysv64" fn struct_24_bits(a: S24) -> S24 {
  a
}

// CHECK: i48 @struct_48_bits(i48
#[no_mangle]
pub extern "sysv64" fn struct_48_bits(a: S48) -> S48 {
  a
}
