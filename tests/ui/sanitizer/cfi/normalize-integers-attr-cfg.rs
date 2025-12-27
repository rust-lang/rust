// Verifies that when compiling with `-Zsanitizer-cfi-normalize-integers` the
// `#[cfg(sanitizer_cfi_normalize_integers)]` attribute is configured.
//
//@ needs-sanitizer-cfi
//@ check-pass
//@ compile-flags: -Ccodegen-units=1 -Clto -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=cfi -Zsanitizer-cfi-normalize-integers

#![feature(cfg_sanitizer_cfi)]

#[cfg(sanitizer_cfi_normalize_integers)]
fn main() {}
