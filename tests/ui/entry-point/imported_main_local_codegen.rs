//@ run-pass
//@ aux-build:main_functions.rs
//@ compile-flags: -Ccodegen-units=1024

// This is a regression test for https://github.com/rust-lang/rust/issues/144052.
// Entrypoint functions call each other in ways that CGU partitioning doesn't know about. So there
// is a special check to not internalize any of them. But internalizing them can be okay if there
// are few enough CGUs, so we use a lot of CGUs in this test to hit the bad case.

extern crate main_functions;
pub use main_functions::local_codegen as main;
