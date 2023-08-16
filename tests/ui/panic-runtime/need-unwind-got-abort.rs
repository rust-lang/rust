// build-fail
//@error-in-other-file:is incompatible with this crate's strategy of `abort`
//@aux-build:needs-unwind.rs
//@compile-flags:-C panic=abort
// no-prefer-dynamic

extern crate needs_unwind;

fn main() {}
