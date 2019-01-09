// compile-flags: -O

#![crate_type="rlib"]

// CHECK-LABEL: @memzero
// CHECK-NOT: store
// CHECK: call void @llvm.memset
// CHECK-NOT: store
#[no_mangle]
pub fn memzero(data: &mut [u8]) {
    for i in 0..data.len() {
        data[i] = 0;
    }
}
