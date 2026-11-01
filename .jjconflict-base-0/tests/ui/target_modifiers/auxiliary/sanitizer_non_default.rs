// This represents an rlib where CFI is explicitly provided as a -Zsanitizer flag.
// CFI is not a default_sanitizer for riscv64gc-unknown-fuchsia.

//@ no-prefer-dynamic
//@ compile-flags: --target riscv64gc-unknown-fuchsia -Zsanitizer=cfi -Clinker-plugin-lto
//@ needs-llvm-components: riscv
//@ ignore-backends: gcc

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
