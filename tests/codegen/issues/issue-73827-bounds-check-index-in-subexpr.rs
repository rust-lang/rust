// This test checks that bounds checks are elided when
// index is part of a (x | y) < C style condition

// compile-flags: -O

#![crate_type = "lib"]

// CHECK-LABEL: @get
#[no_mangle]
pub fn get(array: &[u8; 8], x: usize, y: usize) -> u8 {
    if x > 7 || y > 7 {
        0
    } else {
        // CHECK-NOT: panic_bounds_check
        array[y]
    }
}
