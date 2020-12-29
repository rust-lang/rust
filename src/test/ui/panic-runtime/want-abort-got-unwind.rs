// build-fail
// dont-check-compiler-stderr
// error-pattern:is not compiled with this crate's panic strategy `abort`
// aux-build:panic-runtime-unwind.rs
// compile-flags:-C panic=abort

extern crate panic_runtime_unwind;

fn main() {}
