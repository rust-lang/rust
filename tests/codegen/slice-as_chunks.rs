//@ compile-flags: -O
//@ only-64bit (because the LLVM type of i64 for usize shows up)

#![crate_type = "lib"]
#![feature(slice_as_chunks)]

// CHECK-LABEL: @chunks4
#[no_mangle]
pub fn chunks4(x: &[u8]) -> &[[u8; 4]] {
    // CHECK-NOT: shl
    // CHECK-NOT: mul
    // CHECK-NOT: udiv
    // CHECK-NOT: urem
    // CHECK: %[[NEWLEN:.+]] = lshr i64 %x.1, 2
    // CHECK: %[[A:.+]] = insertvalue { ptr, i64 } poison, ptr %x.0, 0
    // CHECK: %[[B:.+]] = insertvalue { ptr, i64 } %[[A]], i64 %[[NEWLEN]], 1
    // CHECK: ret { ptr, i64 } %[[B]]
    x.as_chunks().0
}

// CHECK-LABEL: @chunks4_with_remainder
#[no_mangle]
pub fn chunks4_with_remainder(x: &[u8]) -> (&[[u8; 4]], &[u8]) {
    // CHECK-DAG: and i64 %x.1, [[#0x7FFFFFFFFFFFFFFC]]
    // CHECK-DAG: and i64 %x.1, 3
    // CHECK-DAG: lshr i64 %x.1, 2
    // CHECK-NOT: mul
    // CHECK-NOT: udiv
    // CHECK-NOT: urem
    // CHECK: ret
    x.as_chunks()
}
