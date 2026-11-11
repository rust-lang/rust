//@ compile-flags: -Z sanitizer=kernel-hwaddress -Z sanitizer=kernel-address --target aarch64-unknown-none
//@ needs-llvm-components: aarch64
//@ ignore-backends: gcc

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer=kernel-address` is incompatible with `-Zsanitizer=kernel-hwaddress`
