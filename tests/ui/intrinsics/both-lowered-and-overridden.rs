//! Check that intrinsics that do not get overridden, but are marked as such,
//! cause an error instead of silently invoking the body.
#![feature(intrinsics)]
//@ check-fail
//@ failure-status:101
//@ normalize-stderr-test: ".*note: .*\n\n" -> ""
//@ normalize-stderr-test: "thread 'rustc' panicked.*:\n.*\n" -> ""
//@ normalize-stderr-test: "internal compiler error:.*: " -> ""
//@ error-pattern: intrinsic const_deallocate should be marked with at most one of rustc_intrinsic_must_be_overridden and rustc_intrinsic_lowers_to_mir
//@ rustc-env:RUST_BACKTRACE=0

#[rustc_intrinsic]
#[rustc_intrinsic_lowers_to_mir]
#[rustc_intrinsic_must_be_overridden]
pub const unsafe fn const_deallocate(_ptr: *mut u8, _size: usize, _align: usize) {}

fn main() {
    unsafe { const_deallocate(std::ptr::null_mut(), 0, 0) }
}
