// Verifies that unstable supported sanitizers cannot be used without `-Zunstable-options`.
//
//@ needs-llvm-components: x86
//@ needs-sanitizer-support
//@ ignore-backends: gcc
//@ compile-flags: -Cunsafe-allow-abi-mismatch=sanitize -Csanitize=kernel-address --target x86_64-unknown-none

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR kernel-address sanitizer is not supported for this target
