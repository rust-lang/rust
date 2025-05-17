// Verifies that when compiling with `-Zsanitizer-cfi-generalize-pointers` the
// `#[cfg(sanitizer_cfi_generalize_pointers)]` attribute is configured.
//
//@ needs-sanitizer-cfi
//@ check-pass
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Zsanitizer-cfi-generalize-pointers
//@ compile-flags: -C unsafe-allow-abi-mismatch=sanitizer

#![feature(cfg_sanitizer_cfi)]

#[cfg(sanitizer_cfi_generalize_pointers)]
fn main() {}
