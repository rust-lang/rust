//@ compile-flags: -O -Z merge-functions=disabled
#![crate_type = "lib"]

// This tests that LLVM can optimize based on the niches in the source or
// destination types for casts.

// CHECK-LABEL: @u32_index
#[no_mangle]
pub fn u32_index(c: u32) -> [bool; 10] {
    let mut array = [false; 10];

    let index = (c | 1).leading_zeros() as usize / 4 - 2;

    // CHECK: call core::panicking::panic
    array[index as usize] = true;

    array
}

// CHECK-LABEL: @char_as_u32_index
#[no_mangle]
pub fn char_as_u32_index(c: char) -> [bool; 10] {
    let c = c as u32;

    let mut array = [false; 10];

    let index = (c | 1).leading_zeros() as usize / 4 - 2;

    // CHECK-NOT: call core::panicking::panic
    array[index as usize] = true;

    array
}
