//@ build-fail
//@ aux-build:needs-unwind.rs
//@ compile-flags:-C panic=abort
//@ no-prefer-dynamic

extern crate needs_unwind;

fn main() {}

//~? ERROR the crate `needs_unwind` requires panic strategy `unwind` which is incompatible with this crate's strategy of `abort`
