//@ compile-flags: -O
#![crate_type = "lib"]

//! Check that pointer flowing into vec.append comes straight from an allocation
//! and not through a phi node that merges the allocation and zero-length cases.
//! With this and https://github.com/llvm/llvm-project/pull/110280 the intermediate
//! allocation should be optimized away in the future.

// CHECK-LABEL: @vec_append_with_temp_alloc
#[no_mangle]
pub fn vec_append_with_temp_alloc(dst: &mut Vec<u8>, src: &[u8]) {
    // CHECK: %[[TEMP:.+]] = tail call noalias noundef ptr @__rust_alloc

    // First memcpy, it uses the src pointer directly
    // CHECK: call void @llvm.memcpy.{{.*}}%src.0
    let temp = src.to_vec();

    // final memcpy to destination
    // CHECK: call void @llvm.memcpy.{{.*}}%dst.i{{.*}}%[[TEMP]]
    dst.extend(&temp);
    // CHECK: ret
}

// CHECK-LABEL: @string_append_with_temp_alloc
#[no_mangle]
pub fn string_append_with_temp_alloc(dst: &mut String, src: &str) {
    // CHECK: %[[TEMP:.+]] = tail call noalias noundef ptr @__rust_alloc

    // First memcpy, it uses the src pointer directly
    // CHECK: call void @llvm.memcpy.{{.*}}%src.0
    let temp = src.to_string();

    // final memcpy to destination
    // CHECK: call void @llvm.memcpy.{{.*}}%dst.i{{.*}}%[[TEMP]]
    dst.push_str(&temp);
    // CHECK: ret
}
