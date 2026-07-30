//@ compile-flags: -T sanitizer=kernel-hwaddress -T sanitizer=kernel-address --target aarch64-unknown-none
//@ compile-flags: -Z unstable-options
//@ needs-llvm-components: aarch64
//@ ignore-backends: gcc

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Tsanitizer=kernel-address` is incompatible with `-Tsanitizer=kernel-hwaddress`
