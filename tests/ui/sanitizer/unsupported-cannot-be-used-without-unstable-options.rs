// Verifies that unsupported sanitizers cannot be used without `-Zunstable-options`.
//
//@ needs-llvm-components: x86
//@ compile-flags: -Csanitize=kernel-address --target x86_64-unknown-linux-gnu
//@ error-pattern: error: kernel-address sanitizer is not supported for this target

#![feature(no_core)]
#![no_core]
#![no_main]
