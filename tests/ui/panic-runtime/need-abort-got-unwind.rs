//@ build-fail
//@ needs-unwind
//@ aux-build:needs-abort.rs

extern crate needs_abort;

fn main() {}

//~? ERROR the crate `needs_abort` requires panic strategy `abort` which is incompatible with this crate's strategy of `unwind`
