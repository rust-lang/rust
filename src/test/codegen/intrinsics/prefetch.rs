// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::{prefetch_read_data, prefetch_write_data,
                      prefetch_read_instruction, prefetch_write_instruction};

#[no_mangle]
pub fn check_prefetch_read_data(data: &[i8]) {
    unsafe {
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 0, i32 0, i32 1)
        prefetch_read_data(data.as_ptr(), 0);
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 0, i32 1, i32 1)
        prefetch_read_data(data.as_ptr(), 1);
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 0, i32 2, i32 1)
        prefetch_read_data(data.as_ptr(), 2);
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 0, i32 3, i32 1)
        prefetch_read_data(data.as_ptr(), 3);
    }
}

#[no_mangle]
pub fn check_prefetch_write_data(data: &[i8]) {
    unsafe {
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 1, i32 0, i32 1)
        prefetch_write_data(data.as_ptr(), 0);
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 1, i32 1, i32 1)
        prefetch_write_data(data.as_ptr(), 1);
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 1, i32 2, i32 1)
        prefetch_write_data(data.as_ptr(), 2);
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 1, i32 3, i32 1)
        prefetch_write_data(data.as_ptr(), 3);
    }
}

#[no_mangle]
pub fn check_prefetch_read_instruction(data: &[i8]) {
    unsafe {
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 0, i32 0, i32 0)
        prefetch_read_instruction(data.as_ptr(), 0);
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 0, i32 1, i32 0)
        prefetch_read_instruction(data.as_ptr(), 1);
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 0, i32 2, i32 0)
        prefetch_read_instruction(data.as_ptr(), 2);
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 0, i32 3, i32 0)
        prefetch_read_instruction(data.as_ptr(), 3);
    }
}

#[no_mangle]
pub fn check_prefetch_write_instruction(data: &[i8]) {
    unsafe {
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 1, i32 0, i32 0)
        prefetch_write_instruction(data.as_ptr(), 0);
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 1, i32 1, i32 0)
        prefetch_write_instruction(data.as_ptr(), 1);
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 1, i32 2, i32 0)
        prefetch_write_instruction(data.as_ptr(), 2);
        // CHECK: call void @llvm.prefetch(i8* %{{.*}}, i32 1, i32 3, i32 0)
        prefetch_write_instruction(data.as_ptr(), 3);
    }
}
