// Test linking using `cc` with `rust-lld`, using a custom target with features described in MCP 510
// see https://github.com/rust-lang/compiler-team/issues/510 for more info:
//
// Starting from the `x86_64-unknown-linux-gnu` target spec, we add the following options:
// - a linker-flavor using lld via a C compiler
// - the self-contained linker component is enabled

#![feature(no_core)]
#![no_core]
