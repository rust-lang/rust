//@ compile-flags: -Ctarget-feature=+rdrnd --crate-type=rlib --target=x86_64-unknown-linux-gnu
//@ build-pass
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]

//~? WARN unknown and unstable feature specified for `-Ctarget-feature`: `rdrnd`
