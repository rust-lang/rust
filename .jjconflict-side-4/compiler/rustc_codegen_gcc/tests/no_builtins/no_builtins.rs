// Test that the #![no_builtins] attribute is honored.
// When this attribute is present, GCC should not replace code patterns
// (like loops) with calls to builtins (like memset).
// See https://github.com/rust-lang/rustc_codegen_gcc/issues/570
//
// This test is verified by the build system test `--no-builtins-tests` which
// compiles this file and checks that `memset` is not referenced in the object file.

#![no_std]
#![no_builtins]
#![crate_type = "lib"]

// This function implements a byte-setting loop that GCC would typically
// optimize into a memset call. With #![no_builtins], GCC should preserve
// the loop instead of replacing it with a builtin call.
#[no_mangle]
#[inline(never)]
pub unsafe fn set_bytes(mut s: *mut u8, c: u8, n: usize) {
    let end = s.add(n);
    while s < end {
        *s = c;
        s = s.add(1);
    }
}
