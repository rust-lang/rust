//@ compile-flags: -Csanitizer=leak --target i686-unknown-linux-gnu -Zunstable-options
//@ needs-llvm-components: x86
//@ ignore-backends: gcc

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR leak sanitizer is not supported for this target
