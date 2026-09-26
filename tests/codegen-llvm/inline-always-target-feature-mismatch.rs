//@ only-x86_64
//@ min-llvm-version: 23
//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]
#![feature(test)]
#![allow(unused_unsafe)]

use std::arch::x86_64::_bzhi_u64;

extern crate test;
use test::black_box as b;

#[inline(always)]
#[target_feature(enable = "bmi2")]
#[unsafe(no_mangle)]
pub unsafe fn callee_requires_bmi2() -> u64 {
    // black box this as `_bzhi_u64(1, 2)` can evaluate to `1` at compile time and LLVM is smart
    // enough to see that it then can ignore `bmi2` as this returns a constant. Which is safe to
    // do.
    b(_bzhi_u64(1, 2))
}

#[unsafe(no_mangle)]
// CHECK-LABEL: define{{.*}} @caller_only()
// CHECK: [[TMP:%.+]] = tail call noundef i64 @callee_requires_bmi2()
pub unsafe fn caller_only() {
    let _x = callee_requires_bmi2();
}
