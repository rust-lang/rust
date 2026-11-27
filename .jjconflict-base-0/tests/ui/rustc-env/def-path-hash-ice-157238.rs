//@ run-pass
//
// Ensure that `trace` level instrumentation in `def_path_hash` works,
// see #157238.
//
//@ dont-check-compiler-stdout
//@ dont-check-compiler-stderr
//@ rustc-env:RUSTC_LOG=trace
fn main() {}
