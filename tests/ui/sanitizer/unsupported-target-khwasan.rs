//@ compile-flags: -Tsanitizer=kernel-hwaddress --target x86_64-unknown-none -Zunstable-options
//@ needs-llvm-components: x86
//@ ignore-backends: gcc

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR kernel-hwaddress sanitizer is not supported for this target
