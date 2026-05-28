//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::{
    prefetch_read_data, prefetch_read_instruction, prefetch_write_data, prefetch_write_instruction,
};

#[no_mangle]
pub fn check_prefetch_read_data(data: &[i8]) {
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 0, i32 0, i32 1)
    prefetch_read_data::<_, 0>(data.as_ptr());
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 0, i32 1, i32 1)
    prefetch_read_data::<_, 1>(data.as_ptr());
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 0, i32 2, i32 1)
    prefetch_read_data::<_, 2>(data.as_ptr());
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 0, i32 3, i32 1)
    prefetch_read_data::<_, 3>(data.as_ptr());
}

#[no_mangle]
pub fn check_prefetch_write_data(data: &[i8]) {
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 1, i32 0, i32 1)
    prefetch_write_data::<_, 0>(data.as_ptr());
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 1, i32 1, i32 1)
    prefetch_write_data::<_, 1>(data.as_ptr());
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 1, i32 2, i32 1)
    prefetch_write_data::<_, 2>(data.as_ptr());
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 1, i32 3, i32 1)
    prefetch_write_data::<_, 3>(data.as_ptr());
}

#[no_mangle]
pub fn check_prefetch_read_instruction(data: &[i8]) {
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 0, i32 0, i32 0)
    prefetch_read_instruction::<_, 0>(data.as_ptr());
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 0, i32 1, i32 0)
    prefetch_read_instruction::<_, 1>(data.as_ptr());
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 0, i32 2, i32 0)
    prefetch_read_instruction::<_, 2>(data.as_ptr());
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 0, i32 3, i32 0)
    prefetch_read_instruction::<_, 3>(data.as_ptr());
}

#[no_mangle]
pub fn check_prefetch_write_instruction(data: &[i8]) {
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 1, i32 0, i32 0)
    prefetch_write_instruction::<_, 0>(data.as_ptr());
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 1, i32 1, i32 0)
    prefetch_write_instruction::<_, 1>(data.as_ptr());
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 1, i32 2, i32 0)
    prefetch_write_instruction::<_, 2>(data.as_ptr());
    // CHECK: call void @llvm.prefetch{{.*}}({{.*}}, i32 1, i32 3, i32 0)
    prefetch_write_instruction::<_, 3>(data.as_ptr());
}
