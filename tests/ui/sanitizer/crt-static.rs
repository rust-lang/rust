//@ compile-flags: -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=address -Ctarget-feature=+crt-static --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR sanitizer is incompatible with statically linked libc
