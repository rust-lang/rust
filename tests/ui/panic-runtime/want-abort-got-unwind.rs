//@ build-fail
//@ dont-check-compiler-stderr
//@ aux-build:panic-runtime-unwind.rs
//@ compile-flags:-C panic=abort

extern crate panic_runtime_unwind;

fn main() {}

//~? ERROR cannot link together two panic runtimes: panic_unwind and panic_runtime_unwind
//~? ERROR the linked panic runtime `panic_runtime_unwind` is not compiled with this crate's panic strategy `abort`
//~? ERROR the crate `panic_unwind` requires panic strategy `unwind` which is incompatible with this crate's strategy of `abort`
