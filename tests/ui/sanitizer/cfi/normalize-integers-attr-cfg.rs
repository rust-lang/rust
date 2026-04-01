// Verifies that when compiling with `-Zsanitizer-cfi-normalize-integers` the
// `#[cfg(sanitizer_cfi_normalize_integers)]` attribute is configured.
//
//@ needs-sanitizer-cfi
//@ check-pass
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Zsanitizer-cfi-normalize-integers -C unsafe-allow-abi-mismatch=sanitizer,sanitizer-cfi-normalize-integers

#![feature(cfg_sanitizer_cfi)]

#[cfg(sanitizer_cfi_normalize_integers)]
fn main() {}
