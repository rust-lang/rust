// ignore-tidy-linelength
//@ build-fail
//@ compile-flags: --error-format=human
//@ error-pattern: the linked panic runtime `panic_runtime_unwind` is not compiled with this crate's panic strategy `abort`
//@ dont-check-compiler-stderr
//@ aux-build:panic-runtime-unwind.rs
//@ compile-flags:-C panic=abort

extern crate panic_runtime_unwind;

fn main() {}

// FIXME: The first and third errors are target-dependent.
//FIXME~? ERROR cannot link together two panic runtimes: panic_unwind and panic_runtime_unwind
//FIXME~? ERROR the linked panic runtime `panic_runtime_unwind` is not compiled with this crate's panic strategy `abort`
//FIXME~? ERROR the crate `panic_unwind` requires panic strategy `unwind` which is incompatible with this crate's strategy of `abort`
