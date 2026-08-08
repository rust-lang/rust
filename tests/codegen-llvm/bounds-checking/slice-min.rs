//! Regression test for #<https://github.com/rust-lang/rust/issues/120433>:
//! Multiple bounds checking elision failures
//! (ensures bounds checks are properly elided,
//! with no calls to panic_bounds_check in the LLVM IR).

//@ compile-flags: -C opt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @foo
// CHECK-NOT: panic_bounds_check
#[no_mangle]
pub fn foo(buf: &[u8], alloced_size: usize) -> &[u8] {
    if alloced_size.checked_add(1).map(|total| buf.len() < total).unwrap_or(true) {
        return &[];
    }
    let size = buf[0];
    &buf[1..1 + usize::min(alloced_size, usize::from(size))]
}
