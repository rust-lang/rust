//@ compile-flags: -Ctarget-feature=rdrand --crate-type=rlib --target=x86_64-unknown-linux-gnu
//@ build-pass
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_core]

//~? WARN ignoring feature with missing prefix in `-Ctarget-feature`: `rdrand`
