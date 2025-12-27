// Verifies that invalid user-defined CFI encodings can't be used.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Ccodegen-units=1 -Clto -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=cfi -Zsanitizer-cfi-generalize-pointers

#![feature(cfi_encoding, no_core)]
#![no_core]
#![no_main]

#[cfi_encoding] //~ ERROR malformed `cfi_encoding` attribute input
pub struct Type1(i32);
