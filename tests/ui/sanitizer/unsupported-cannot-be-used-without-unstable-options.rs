// Verifies that unsupported sanitizers cannot be used without `-Zunstable-options`.
//
//@ needs-llvm-components: x86
//@ compile-flags: -Cunsafe-allow-abi-mismatch=sanitize -Csanitize=kernel-address --target x86_64-unknown-linux-gnu

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR kernel-address sanitizer is not supported for this target
