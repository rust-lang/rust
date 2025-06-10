// Verifies that `#![no_builtins]` crates can be built with linker-plugin-lto and CFI.
// See Issue #142284
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clinker-plugin-lto -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static
//@ compile-flags: --crate-type rlib
//@ build-pass

#![no_builtins]
#![no_std]

pub static FUNC: fn() = initializer;

pub fn initializer() {
    call(fma_with_fma);
}

pub fn call(fn_ptr: fn()) {
    fn_ptr();
}

pub fn fma_with_fma() {}
