// Ensure the asm for array comparisons is properly optimized.

//@ compile-flags: -C opt-level=2

#![crate_type = "lib"]

// CHECK-LABEL: @compare
// CHECK: start:
// CHECK-NEXT: ret i1 true
#[no_mangle]
pub fn compare() -> bool {
    let bytes = 12.5f32.to_ne_bytes();
    bytes
        == if cfg!(target_endian = "big") {
            [0x41, 0x48, 0x00, 0x00]
        } else {
            [0x00, 0x00, 0x48, 0x41]
        }
}

// CHECK-LABEL: @array_of_tuple_le
// CHECK: call{{.+}}i8 @llvm.scmp.i8.i16
// CHECK: call{{.+}}i8 @llvm.ucmp.i8.i16
// CHECK: call{{.+}}i8 @llvm.scmp.i8.i16
// CHECK: call{{.+}}i8 @llvm.ucmp.i8.i16
// CHECK: %[[RET:.+]] = icmp slt i8 {{.+}}, 1
// CHECK: ret i8 %[[RET]]
#[no_mangle]
pub fn array_of_tuple_le(a: &[(i16, u16); 2], b: &[(i16, u16); 2]) -> bool {
    a <= b
}
