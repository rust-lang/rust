//@ compile-flags: --target=x86_64-unknown-linux-gnu -C linker-flavor=msvc --crate-type=rlib
//@ needs-llvm-components:

#![feature(no_core)]
#![no_core]

//~? ERROR linker flavor `msvc` is incompatible with the current target
