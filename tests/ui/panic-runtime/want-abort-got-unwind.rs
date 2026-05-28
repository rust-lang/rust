// ignore-tidy-linelength
//@ build-fail
//@ dont-check-compiler-stderr
//@ aux-build:panic-runtime-unwind.rs
//@ compile-flags:-C panic=abort

// NOTE: depending on the target's default panic strategy, there can be additional errors that
// complain about linking two panic runtimes (e.g. precompiled `panic_unwind` if target default
// panic strategy is unwind, in addition to `panic_runtime_unwind`). These additional errors will
// not be observed on targets whose default panic strategy is abort, where `panic_abort` is linked
// in instead.
//@ dont-require-annotations: ERROR

extern crate panic_runtime_unwind;

fn main() {}

//~? ERROR the linked panic runtime `panic_runtime_unwind` is not compiled with this crate's panic strategy `abort`
