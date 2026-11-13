//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

// Ensure that a modulo operation with an operand that is known to be
// a power-of-two is properly optimized.

// CHECK-LABEL: @modulo_with_power_of_two_divisor
// CHECK: add i64 %divisor, -1
// CHECK-NEXT: and i64
// CHECK-NEXT: ret i64
#[no_mangle]
pub fn modulo_with_power_of_two_divisor(dividend: u64, divisor: u64) -> u64 {
    assert!(divisor.is_power_of_two());
    // should be optimized to (dividend & (divisor - 1))
    dividend % divisor
}
