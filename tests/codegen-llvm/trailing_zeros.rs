//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @trailing_zeros_ge
#[no_mangle]
pub fn trailing_zeros_ge(val: u32) -> bool {
    // CHECK: %[[AND:.*]] = and i32 %val, 7
    // CHECK: %[[ICMP:.*]] = icmp eq i32 %[[AND]], 0
    // CHECK: ret i1 %[[ICMP]]
    val.trailing_zeros() >= 3
}

// CHECK-LABEL: @trailing_zeros_gt
#[no_mangle]
pub fn trailing_zeros_gt(val: u64) -> bool {
    // CHECK: %[[AND:.*]] = and i64 %val, 15
    // CHECK: %[[ICMP:.*]] = icmp eq i64 %[[AND]], 0
    // CHECK: ret i1 %[[ICMP]]
    val.trailing_zeros() > 3
}
