// Matching every possible result of `x % 5` should not keep unreachable
// fallback panic code.

//@ compile-flags: -Copt-level=3
//@ only-64bit

#![crate_type = "lib"]

unsafe extern "C" {
    fn rem_arm0(x: usize) -> usize;
    fn rem_arm1(x: usize) -> usize;
    fn rem_arm2(x: usize) -> usize;
    fn rem_arm3(x: usize) -> usize;
    fn rem_arm4(x: usize) -> usize;
}

// CHECK-LABEL: @rem_match_unreachable
// CHECK-NOT: core::panicking::panic
// CHECK-NOT: entered unreachable code
#[no_mangle]
pub fn rem_match_unreachable(x: usize) -> usize {
    unsafe {
        match x % 5 {
            0 => rem_arm0(x),
            1 => rem_arm1(x),
            2 => rem_arm2(x),
            3 => rem_arm3(x),
            4 => rem_arm4(x),
            _ => unreachable!(),
        }
    }
}
