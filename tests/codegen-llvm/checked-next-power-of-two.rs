//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]
#![no_std]

// CHECK-LABEL: @checked_next_power_of_two_properties
// CHECK: ret i1 true
#[no_mangle]
pub fn checked_next_power_of_two_properties(value: u64) -> bool {
    value
        .checked_next_power_of_two()
        .is_none_or(|result| result.is_power_of_two() && result >= value)
}

// CHECK-LABEL: @modulo_checked_next_power_of_two
// CHECK-NOT: udiv
// CHECK-NOT: urem
// CHECK: and i64
// CHECK-NOT: udiv
// CHECK-NOT: urem
// CHECK: ret
#[no_mangle]
pub fn modulo_checked_next_power_of_two(value: u64, dividend: u64) -> Option<u64> {
    value.checked_next_power_of_two().map(|divisor| dividend % divisor)
}
