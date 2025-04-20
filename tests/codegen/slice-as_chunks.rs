//@ compile-flags: -Copt-level=3
//@ only-64bit (because the LLVM type of i64 for usize shows up)

#![crate_type = "lib"]

// CHECK-LABEL: @chunks4
#[no_mangle]
pub fn chunks4(x: &[u8]) -> &[[u8; 4]] {
    // CHECK-NEXT: start:
    // CHECK-NEXT: lshr i64 %x.1, 2
    // CHECK-NOT: shl
    // CHECK-NOT: mul
    // CHECK-NOT: udiv
    // CHECK-NOT: urem
    // CHECK: ret
    x.as_chunks().0
}

// CHECK-LABEL: @chunks4_with_remainder
#[no_mangle]
pub fn chunks4_with_remainder(x: &[u8]) -> (&[[u8; 4]], &[u8]) {
    // CHECK-DAG: and i64 %x.1, -4
    // CHECK-DAG: and i64 %x.1, 3
    // CHECK-DAG: lshr
    // CHECK-NOT: mul
    // CHECK-NOT: udiv
    // CHECK-NOT: urem
    // CHECK: ret
    x.as_chunks()
}
