// build-fail
// needs-unwind
// error-pattern:is incompatible with this crate's strategy of `unwind`
// aux-build:needs-abort.rs

extern crate needs_abort;

fn main() {}
