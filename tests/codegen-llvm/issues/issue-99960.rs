//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[no_mangle]
pub fn test(dividend: i64, divisor: i64) -> Option<i64> {
    // CHECK-LABEL: @test(
    // CHECK-NOT: panic
    if dividend > i64::min_value() && divisor != 0 { Some(dividend / divisor) } else { None }
}
