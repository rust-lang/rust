// run-pass
// This test is just checking that we won't ICE if logging is turned
// on; don't bother trying to compare that (copious) output.
//
// dont-check-compiler-stdout
// dont-check-compiler-stderr
// compile-flags: --error-format human
// aux-build: rustc-rust-log-aux.rs
// rustc-env:RUSTC_LOG=debug

fn main() {}
