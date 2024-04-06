//@ compile-flags: -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=kernel-address -Csanitize=kernel-hwaddress --target aarch64-unknown-none
//@ needs-llvm-components: aarch64
//@ ignore-backends: gcc

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Csanitize=kernel-address` is incompatible with `-Csanitize=kernel-hwaddress`
