// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
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

// CHECK-LABEL: @zero_sized_elem
#[no_mangle]
pub fn zero_sized_elem() {
    // CHECK-NOT: br label %repeat_loop_header{{.*}}
    // CHECK-NOT: call void @llvm.memset.p0i8
    let x = [(); 4];
    drop(&x);
}

// CHECK-LABEL: @zero_len_array
#[no_mangle]
pub fn zero_len_array() {
    // CHECK-NOT: br label %repeat_loop_header{{.*}}
    // CHECK-NOT: call void @llvm.memset.p0i8
    let x = [4; 0];
    drop(&x);
}

// CHECK-LABEL: @byte_array
#[no_mangle]
pub fn byte_array() {
    // CHECK: call void @llvm.memset.p0i8.i[[WIDTH:[0-9]+]](i8* {{.*}}, i8 7, i[[WIDTH]] 4
    // CHECK-NOT: br label %repeat_loop_header{{.*}}
    let x = [7u8; 4];
    drop(&x);
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
enum Init {
    Loop,
    Memset,
}

// CHECK-LABEL: @byte_enum_array
#[no_mangle]
pub fn byte_enum_array() {
    // CHECK: call void @llvm.memset.p0i8.i[[WIDTH:[0-9]+]](i8* {{.*}}, i8 {{.*}}, i[[WIDTH]] 4
    // CHECK-NOT: br label %repeat_loop_header{{.*}}
    let x = [Init::Memset; 4];
    drop(&x);
}

// CHECK-LABEL: @zeroed_integer_array
#[no_mangle]
pub fn zeroed_integer_array() {
    // CHECK: call void @llvm.memset.p0i8.i[[WIDTH:[0-9]+]](i8* {{.*}}, i8 0, i[[WIDTH]] 16
    // CHECK-NOT: br label %repeat_loop_header{{.*}}
    let x = [0u32; 4];
    drop(&x);
}

// CHECK-LABEL: @nonzero_integer_array
#[no_mangle]
pub fn nonzero_integer_array() {
    // CHECK: br label %repeat_loop_header{{.*}}
    // CHECK-NOT: call void @llvm.memset.p0i8
    let x = [0x1a_2b_3c_4d_u32; 4];
    drop(&x);
}
