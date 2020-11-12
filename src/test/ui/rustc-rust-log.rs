// run-pass
// This test is just checking that we won't ICE if logging is turned
// on; don't bother trying to compare that (copious) output. (Note
// also that this test potentially silly, since we do not build+test
// debug versions of rustc as part of our continuous integration
// process...)
//
// dont-check-compiler-stdout
// dont-check-compiler-stderr
// compile-flags: --error-format human
// aux-build: rustc-rust-log-aux.rs
// rustc-env:RUSTC_LOG=debug

fn main() {}
