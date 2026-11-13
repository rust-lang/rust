//@ compile-flags: --target aarch64-unknown-linux-pauthtest -Zpointer-authentication=+calls,+init-fini
//@ needs-llvm-components: aarch64
//@ no-prefer-dynamic
//@ ignore-backends: gcc

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
