//@ no-prefer-dynamic
//@ compile-flags: --target x86_64-unknown-linux-gnu -Zllvm-target-feature=+fake-feature
//@ needs-llvm-components: x86

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
