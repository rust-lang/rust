//@ revisions: opt-3 opt-z
//@[opt-3] compile-flags: -Copt-level=3
//@[opt-z] compile-flags: -Copt-level=z
// Regression test for https://github.com/rust-lang/rust/issues/152061.
//
// `impl fmt::Display` for integers, lowered through `_fmt_inner` in
// `library/core/src/fmt/num.rs`, used to leave a `panic_bounds_check`
// path in optimised LLVM IR when LLVM failed to propagate the
// `assume`-based range information (notably under `opt-level=z` + fat
// LTO with LLVM 21). The implementation was rewritten to use
// `get_unchecked{_mut}` for the buffer writes, so the `panic_bounds_check`
// path must not appear regardless of the optimiser's propagation.

#![crate_type = "lib"]

use std::fmt::Write;

pub struct NoopWriter;
impl Write for NoopWriter {
    fn write_str(&mut self, _s: &str) -> std::fmt::Result {
        Ok(())
    }
}

// CHECK-LABEL: @format_usize
#[no_mangle]
pub fn format_usize(w: &mut NoopWriter, x: usize) {
    // The Display path through `_fmt_inner` must not emit a bounds check.
    // CHECK-NOT: panic_bounds_check
    let _ = write!(w, "{}", x);
}

// CHECK-LABEL: @format_u64
#[no_mangle]
pub fn format_u64(w: &mut NoopWriter, x: u64) {
    // CHECK-NOT: panic_bounds_check
    let _ = write!(w, "{}", x);
}

// Sanity check: make sure `panic_bounds_check` is still the symbol LLVM
// emits for a non-elidable out-of-bounds index, so the `CHECK-NOT`s
// above are guarding against something real and cannot pass vacuously.
// CHECK-LABEL: @test_check
#[no_mangle]
pub fn test_check(arr: &[u8], i: usize) -> u8 {
    // CHECK: panic_bounds_check
    arr[i]
}
