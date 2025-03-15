//@ compile-flags: -Zunstable-options -Csanitize=address -Csanitize=memory --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86
//@ error-pattern: error: `-Csanitize=address` is incompatible with `-Csanitize=memory`

#![feature(no_core)]
#![no_core]
#![no_main]
