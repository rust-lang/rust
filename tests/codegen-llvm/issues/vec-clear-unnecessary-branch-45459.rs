// Tests that compare and branch are optimized out when clearing a `Vec`, fixed
// since  1.30.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @check_no_compare
#[no_mangle]
pub fn check_no_compare(v: &mut Vec<f32>) {
    // CHECK-NOT: icmp
    // CHECK-NOT: br
    v.clear();
}
