// Verifies that stable and unstable supported sanitizers can be used with `-Zunstable-options`.
//
//@ needs-llvm-components: x86
//@ needs-sanitizer-support
//@ build-pass
//@ compile-flags: -Zunstable-options -Clto -Csanitize=address,cfi --target x86_64-unknown-linux-gnu

#![crate_type = "rlib"]
#![feature(no_core)]
#![no_core]
#![no_main]
