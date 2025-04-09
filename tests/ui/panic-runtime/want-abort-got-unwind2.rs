// ignore-tidy-linelength
//@ build-fail
//@ dont-require-annotations:ERROR
//@ dont-check-compiler-stderr
//@ aux-build:panic-runtime-unwind.rs
//@ aux-build:wants-panic-runtime-unwind.rs
//@ compile-flags:-C panic=abort

extern crate wants_panic_runtime_unwind;

fn main() {}

//~? ERROR the linked panic runtime `panic_runtime_unwind` is not compiled with this crate's panic strategy `abort`
// FIXME: These errors are target-dependent, could be served by some "optional error" annotation
// instead of `dont-require-annotations`.
//FIXME~? ERROR cannot link together two panic runtimes: panic_unwind and panic_runtime_unwind
//FIXME~? ERROR the crate `panic_unwind` requires panic strategy `unwind` which is incompatible with this crate's strategy of `abort`
