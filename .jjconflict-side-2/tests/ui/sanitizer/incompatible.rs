//@ compile-flags: -Z sanitizer=address -Z sanitizer=memory --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer=address` is incompatible with `-Zsanitizer=memory`
