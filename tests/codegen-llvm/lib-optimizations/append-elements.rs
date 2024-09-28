//@ compile-flags: -O -Zmerge-functions=disabled
//@ min-llvm-version: 21
#![crate_type = "lib"]

//! Check that a temporary intermediate allocations can eliminated and replaced
//! with memcpy forwarding.
//! This requires Vec code to be structured in a way that avoids phi nodes from the
//! zero-capacity length flowing into the memcpy arguments.

// CHECK-LABEL: @vec_append_with_temp_alloc
#[no_mangle]
pub fn vec_append_with_temp_alloc(dst: &mut Vec<u8>, src: &[u8]) {
    // CHECK-NOT: call void @llvm.memcpy
    // CHECK: call void @llvm.memcpy.{{.*}}%dst.i{{.*}}%src.0
    // CHECK-NOT: call void @llvm.memcpy
    let temp = src.to_vec();
    dst.extend(&temp);
    // CHECK: ret
}

// CHECK-LABEL: @string_append_with_temp_alloc
#[no_mangle]
pub fn string_append_with_temp_alloc(dst: &mut String, src: &str) {
    // CHECK-NOT: call void @llvm.memcpy
    // CHECK: call void @llvm.memcpy.{{.*}}%dst.i{{.*}}%src.0
    // CHECK-NOT: call void @llvm.memcpy
    let temp = src.to_string();
    dst.push_str(&temp);
    // CHECK: ret
}
