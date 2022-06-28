// build-fail
// needs-unwind
// error-pattern:is incompatible with this crate's strategy of `unwind`
// aux-build:needs-abort.rs
// ignore-wasm32-bare compiled with panic=abort by default

extern crate needs_abort;

fn main() {}
