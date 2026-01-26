//@ revisions: X86_64 LA64
//@ assembly-output: emit-asm
//@ compile-flags: -C opt-level=3
//
//@ [X86_64] only-x86_64
//@ [X86_64] compile-flags: -C target-cpu=znver4
//@ [X86_64] compile-flags: -C llvm-args=-x86-asm-syntax=intel
//
//@ [LA64] only-loongarch64

#![crate_type = "lib"]

/// Verify `is_ascii` generates efficient code on different architectures:
///
/// - x86_64: Must NOT use `kshiftrd`/`kshiftrq` (broken AVX-512 auto-vectorization).
///   The fix uses explicit SSE2 intrinsics (`pmovmskb`/`vpmovmskb`).
///   See: https://github.com/llvm/llvm-project/issues/176906
///
/// - loongarch64: Should use `vmskltz.b` instruction for the fast-path.
///   This architecture still relies on LLVM auto-vectorization.

// X86_64-LABEL: test_is_ascii
// X86_64-NOT: kshiftrd
// X86_64-NOT: kshiftrq

// LA64-LABEL: test_is_ascii
// LA64: vmskltz.b

#[no_mangle]
pub fn test_is_ascii(s: &[u8]) -> bool {
    s.is_ascii()
}
