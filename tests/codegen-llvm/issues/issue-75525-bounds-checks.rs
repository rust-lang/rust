// Regression test for #75525, verifies that no bounds checks are generated.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @f0
// CHECK-NOT: panic
#[no_mangle]
pub fn f0(idx: usize, buf: &[u8; 10]) -> u8 {
    if idx < 8 { buf[idx + 1] } else { 0 }
}

// CHECK-LABEL: @f1
// CHECK-NOT: panic
#[no_mangle]
pub fn f1(idx: usize, buf: &[u8; 10]) -> u8 {
    if idx > 5 && idx < 8 { buf[idx - 1] } else { 0 }
}

// CHECK-LABEL: @f2
// CHECK-NOT: panic
#[no_mangle]
pub fn f2(idx: usize, buf: &[u8; 10]) -> u8 {
    if idx > 5 && idx < 8 { buf[idx] } else { 0 }
}
