//@ compile-flags: -Csanitizer=address -Tsanitizer=memory --target x86_64-unknown-linux-gnu -Zunstable-options
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Zsanitizer=address` is incompatible with `-Tsanitizer=memory`
