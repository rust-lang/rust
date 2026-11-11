// Test that without #![no_builtins], GCC DOES replace code patterns with builtins.
// This is the counterpart to no_builtins.rs - we verify that memset IS emitted
// when the no_builtins attribute is NOT present.
//
// This test is verified by the build system test `--no-builtins-tests` which
// compiles this file and checks that `memset` IS referenced in the object file.

#![no_std]
#![crate_type = "lib"]

// This function implements a byte-setting loop that GCC should optimize
// into a memset call when no_builtins is NOT set.
#[no_mangle]
#[inline(never)]
pub unsafe fn set_bytes(mut s: *mut u8, c: u8, n: usize) {
    let end = s.add(n);
    while s < end {
        *s = c;
        s = s.add(1);
    }
}
