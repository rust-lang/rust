//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// There should be no calls to panic / len_mismatch_fail.

#[no_mangle]
pub fn test(a: &mut [u8], offset: usize, bytes: &[u8]) {
    // CHECK-LABEL: @test(
    // CHECK-NOT: call
    // CHECK: call void @llvm.memcpy
    // CHECK-NOT: call
    // CHECK: }
    if let Some(dst) = a.get_mut(offset..offset + bytes.len()) {
        dst.copy_from_slice(bytes);
    }
}
