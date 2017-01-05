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
pub fn small_array_alignment(x: &mut [i8; 4], y: [i8; 4]) {
// CHECK: [[TMP:%.+]] = alloca i32
// CHECK: %arg1 = alloca [4 x i8]
// CHECK: store i32 %1, i32* [[TMP]]
// CHECK: [[Y8:%[0-9]+]] = bitcast [4 x i8]* %arg1 to i8*
// CHECK: [[TMP8:%[0-9]+]] = bitcast i32* [[TMP]] to i8*
// CHECK: call void @llvm.memcpy.{{.*}}(i8* [[Y8]], i8* [[TMP8]], i{{[0-9]+}} 4, i32 1, i1 false)
    *x = y;
}

// CHECK-LABEL: small_struct_alignment
// The struct is stored as i32, but its alignment is lower, go with 1 byte to avoid target
// dependent alignment
#[no_mangle]
pub fn small_struct_alignment(x: &mut Bytes, y: Bytes) {
// CHECK: [[TMP:%.+]] = alloca i32
// CHECK: %arg1 = alloca %Bytes
// CHECK: store i32 %1, i32* [[TMP]]
// CHECK: [[Y8:%[0-9]+]] = bitcast %Bytes* %arg1 to i8*
// CHECK: [[TMP8:%[0-9]+]] = bitcast i32* [[TMP]] to i8*
// CHECK: call void @llvm.memcpy.{{.*}}(i8* [[Y8]], i8* [[TMP8]], i{{[0-9]+}} 4, i32 1, i1 false)
    *x = y;
}
