//@ only-x86_64
//@ compile-flags: -C opt-level=3 -C target-cpu=znver4
//@ compile-flags: -C llvm-args=-x86-asm-syntax=intel
//@ assembly-output: emit-asm
#![crate_type = "lib"]

// Verify is_ascii uses pmovmskb/vpmovmskb instead of kshiftrd with AVX-512.
// The fix uses explicit SSE2 intrinsics to avoid LLVM's broken auto-vectorization.
//
// See: https://github.com/rust-lang/rust/issues/129293

// CHECK-LABEL: test_is_ascii
#[no_mangle]
pub fn test_is_ascii(s: &[u8]) -> bool {
    // CHECK-NOT: kshiftrd
    // CHECK-NOT: kshiftrq
    s.is_ascii()
}
