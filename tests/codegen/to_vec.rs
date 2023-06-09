// compile-flags: -O

#![crate_type = "lib"]

// CHECK-LABEL: @copy_to_vec
#[no_mangle]
fn copy_to_vec(s: &[u64]) -> Vec<u64> {
  s.to_vec()
  // CHECK: call void @llvm.memcpy
}
