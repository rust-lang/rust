// Test that `wrapping_div` only checks divisor once.
// This test checks that there is only a single compare against -1 and -1 is not present as a
// switch case (the second check present until rustc 1.12).
// This test also verifies that a single panic call is generated (for the division by zero case).

// compile-flags: -O
#![crate_type = "lib"]

// CHECK-LABEL: @f
#[no_mangle]
pub fn f(x: i32, y: i32) -> i32 {
    // CHECK-COUNT-1: icmp eq i32 %y, -1
    // CHECK-COUNT-1: panic
    // CHECK-NOT: i32 -1, label
    x.wrapping_div(y)
}
