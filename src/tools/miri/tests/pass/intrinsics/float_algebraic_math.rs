#![feature(core_intrinsics)]

use std::intrinsics::{
    fadd_algebraic, fdiv_algebraic, fmul_algebraic, frem_algebraic, fsub_algebraic,
};

#[inline(never)]
pub fn test_operations_f64(a: f64, b: f64) {
    // make sure they all map to the correct operation
    assert_eq!(fadd_algebraic(a, b), a + b);
    assert_eq!(fsub_algebraic(a, b), a - b);
    assert_eq!(fmul_algebraic(a, b), a * b);
    assert_eq!(fdiv_algebraic(a, b), a / b);
    assert_eq!(frem_algebraic(a, b), a % b);
}

#[inline(never)]
pub fn test_operations_f32(a: f32, b: f32) {
    // make sure they all map to the correct operation
    assert_eq!(fadd_algebraic(a, b), a + b);
    assert_eq!(fsub_algebraic(a, b), a - b);
    assert_eq!(fmul_algebraic(a, b), a * b);
    assert_eq!(fdiv_algebraic(a, b), a / b);
    assert_eq!(frem_algebraic(a, b), a % b);
}

fn main() {
    test_operations_f64(1., 2.);
    test_operations_f64(10., 5.);
    test_operations_f32(11., 2.);
    test_operations_f32(10., 15.);
}
