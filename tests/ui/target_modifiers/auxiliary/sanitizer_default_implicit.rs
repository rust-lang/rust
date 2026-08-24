// This represents an rlib where SCS is not explicitly provided as a -Zsanitizer flag.
// SCS is a default_sanitizer on riscv64gc-unknown-fuchsia.

//@ no-prefer-dynamic
//@ compile-flags: --target riscv64gc-unknown-fuchsia
//@ needs-llvm-components: riscv
//@ ignore-backends: gcc

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
