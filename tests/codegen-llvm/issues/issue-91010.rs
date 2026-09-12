// Regression test for preserving constant return values after a use of the
// same local through `black_box` or formatting.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::hint::black_box;

// CHECK-LABEL: @black_box_ref_constant
#[no_mangle]
pub fn black_box_ref_constant() -> i32 {
    let x = 1;
    black_box(&x);

    // CHECK: ret i32 1
    x
}

// CHECK-LABEL: @format_ref_constant
#[no_mangle]
pub fn format_ref_constant() -> i32 {
    let x = 1;
    println!("{}", x);

    // CHECK: ret i32 1
    x
}
