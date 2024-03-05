//! Check that intrinsics that do not get overridden, but are marked as such,
//! cause an error instead of silently invoking the body.
#![feature(rustc_attrs, effects)]
//@ build-fail
//@ failure-status:101
//@ normalize-stderr-test ".*note: .*\n\n" -> ""
//@ normalize-stderr-test "thread 'rustc' panicked.*:\n.*\n" -> ""
//@ normalize-stderr-test "internal compiler error:.*: intrinsic const_deallocate " -> ""
//@ rustc-env:RUST_BACKTRACE=0

#[rustc_intrinsic]
#[rustc_intrinsic_must_be_overridden]
pub const unsafe fn const_deallocate(_ptr: *mut u8, _size: usize, _align: usize) {}

fn main() {
    unsafe { const_deallocate(std::ptr::null_mut(), 0, 0) }
    //~^ ERROR: must be overridden
}
