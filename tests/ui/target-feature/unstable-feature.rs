//@ compile-flags: -Ctarget-feature=+vaes --crate-type=rlib --target=x86_64-unknown-linux-gnu
//@ build-pass
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]

//~? WARN unstable feature specified for `-Ctarget-feature`: `vaes`
