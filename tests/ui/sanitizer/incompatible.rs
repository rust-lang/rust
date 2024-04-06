//@ compile-flags: -Zunstable-options -Csanitizer=address -Csanitizer=memory --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86
//@ error-pattern: error: `-Csanitizer=address` is incompatible with `-Csanitizer=memory`

#![feature(no_core)]
#![no_core]
#![no_main]
