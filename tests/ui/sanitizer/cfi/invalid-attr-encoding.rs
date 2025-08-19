// Verifies that invalid user-defined CFI encodings can't be used.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi

#![feature(cfi_encoding, no_core)]
#![no_core]
#![no_main]

#[cfi_encoding] //~ ERROR malformed `cfi_encoding` attribute input
pub struct Type1(i32);
