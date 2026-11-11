//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// Ensure that when val < base, we do not divide or multiply.

// CHECK-LABEL: @checked_ilog
// CHECK-SAME: (i16{{.*}} %val, i16{{.*}} %base)
#[no_mangle]
pub fn checked_ilog(val: u16, base: u16) -> Option<u32> {
    // CHECK-NOT: udiv
    // CHECK-NOT: mul
    // CHECK: %[[IS_LESS:.+]] = icmp ult i16 %val, %base
    // CHECK-NEXT: br i1 %[[IS_LESS]], label %[[TRUE:.+]], label %[[FALSE:.+]]
    // CHECK: [[TRUE]]:
    // CHECK-NOT: udiv
    // CHECK-NOT: mul
    // CHECK: ret { i32, i32 }
    val.checked_ilog(base)
}
