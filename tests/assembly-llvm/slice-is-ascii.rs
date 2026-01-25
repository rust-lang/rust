//@ revisions: LA64
//@ assembly-output: emit-asm
//@ compile-flags: -C opt-level=3
//
//@ [LA64] only-loongarch64

#![crate_type = "lib"]

/// - loongarch64: Should use `vmskltz.b` instruction for the fast-path.

// LA64-LABEL: test_is_ascii
// LA64: vmskltz.b

#[no_mangle]
pub fn test_is_ascii(s: &[u8]) -> bool {
    s.is_ascii()
}
