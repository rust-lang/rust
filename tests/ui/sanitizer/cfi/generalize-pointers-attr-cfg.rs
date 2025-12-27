// Verifies that when compiling with `-Zsanitizer-cfi-generalize-pointers` the
// `#[cfg(sanitizer_cfi_generalize_pointers)]` attribute is configured.
//
//@ needs-sanitizer-cfi
//@ check-pass
//@ compile-flags: -Ccodegen-units=1 -Clto -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=cfi -Zsanitizer-cfi-generalize-pointers

#![feature(cfg_sanitizer_cfi)]

#[cfg(sanitizer_cfi_generalize_pointers)]
fn main() {}
