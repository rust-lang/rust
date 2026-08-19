//@ compile-flags: -C sanitizer=address -C target-feature=+crt-static --target x86_64-unknown-linux-gnu -Z unstable-options
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR sanitizer is incompatible with statically linked libc
