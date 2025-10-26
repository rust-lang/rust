//@ build-fail
//@ aux-build:needs-unwind.rs
//@ compile-flags:-C panic=immediate-abort -Zunstable-options
//@ no-prefer-dynamic

extern crate needs_unwind;

// immediate-abort does not require any panic runtime, so trying to build a binary crate with
// panic=immediate-abort and the precompiled sysroot will fail to link, because no panic runtime
// provides the panic entrypoints used by sysroot crates.
// This test ensures that we get a clean compile error instead of a linker error.

fn main() {}

//~? ERROR the crate `core` was compiled with a panic strategy which is incompatible with `immediate-abort`
