// This represents an rlib where SCS is explicitly provided as a -Zsanitizer flag.

//@ no-prefer-dynamic
//@ compile-flags: --target riscv64gc-unknown-fuchsia -Zsanitizer=shadow-call-stack
//@ needs-llvm-components: riscv
//@ ignore-backends: gcc

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
