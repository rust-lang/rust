//@ compile-flags: -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=address -Csanitize=memory --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR `-Csanitize=address` is incompatible with `-Csanitize=memory`
