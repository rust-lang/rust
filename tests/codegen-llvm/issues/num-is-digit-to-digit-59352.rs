// Tests that there's no panic on unwrapping `to_digit` call after checking
// with `is_digit`.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @num_to_digit_slow
#[no_mangle]
pub fn num_to_digit_slow(num: char) -> u32 {
    // CHECK-NOT: br
    // CHECK-NOT: panic
    if num.is_digit(8) { num.to_digit(8).unwrap() } else { 0 }
}
