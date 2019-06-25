// compile-flags: -O
// ignore-debug: the debug assertions get in the way

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
