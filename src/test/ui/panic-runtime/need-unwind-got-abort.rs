// build-fail
// error-pattern:is incompatible with this crate's strategy of `abort`
// aux-build:needs-unwind.rs
// compile-flags:-C panic=abort

extern crate needs_unwind;

fn main() {}
