// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(rustc_attrs)]

pub struct Bytes {
  a: u8,
  b: u8,
  c: u8,
  d: u8,
}

// CHECK-LABEL: small_array_alignment
// The array is stored as i32, but its alignment is lower, go with 1 byte to avoid target
// dependent alignment
#[no_mangle]
#[rustc_no_mir] // FIXME #27840 MIR has different codegen.
pub fn small_array_alignment(x: &mut [i8; 4], y: [i8; 4]) {
// CHECK: [[VAR:%[0-9]+]] = bitcast [4 x i8]* %y to i32*
// CHECK: store i32 %{{.*}}, i32* [[VAR]], align 1
    *x = y;
}

// CHECK-LABEL: small_struct_alignment
// The struct is stored as i32, but its alignment is lower, go with 1 byte to avoid target
// dependent alignment
#[no_mangle]
#[rustc_no_mir] // FIXME #27840 MIR has different codegen.
pub fn small_struct_alignment(x: &mut Bytes, y: Bytes) {
// CHECK: [[VAR:%[0-9]+]] = bitcast %Bytes* %y to i32*
// CHECK: store i32 %{{.*}}, i32* [[VAR]], align 1
    *x = y;
}
