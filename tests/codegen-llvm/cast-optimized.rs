//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled
#![crate_type = "lib"]

// This tests that LLVM can optimize based on the niches in the source or
// destination types for casts.

// CHECK-LABEL: @u32_index
#[no_mangle]
pub fn u32_index(c: u32) -> [bool; 22] {
    let mut array = [false; 22];

    let index = 32 - c.leading_zeros();

    // CHECK: call core::panicking::panic
    array[index as usize] = true;

    array
}

// CHECK-LABEL: @char_as_u32_index
#[no_mangle]
pub fn char_as_u32_index(c: char) -> [bool; 22] {
    let c = c as u32;

    let mut array = [false; 22];

    let index = 32 - c.leading_zeros();

    // CHECK-NOT: call core::panicking::panic
    array[index as usize] = true;

    array
}
