//! Check that intrinsics that do not get lowered to MIR, but are marked as such,
//! cause an error instead of silently invoking the body.
#![feature(intrinsics)]
//@ build-fail
//@ failure-status:101
//@ normalize-stderr-test: ".*note: .*\n\n" -> ""
//@ normalize-stderr-test: "thread 'rustc' panicked.*:\n.*\n" -> ""
//@ normalize-stderr-test: "internal compiler error:.*: " -> ""
//@ rustc-env:RUST_BACKTRACE=0

#[rustc_intrinsic]
#[rustc_intrinsic_lowers_to_mir]
pub const unsafe fn const_deallocate(_ptr: *mut u8, _size: usize, _align: usize) {}

fn main() {
    unsafe { const_deallocate(std::ptr::null_mut(), 0, 0) }
    //~^ ERROR: Intrinsic const_deallocate was marked as #[rustc_intrinsic_lowers_to_mir] but wasn't lowered by `LowerIntrinsics` pass
}
