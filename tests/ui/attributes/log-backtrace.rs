// run-pass
//
// This test makes sure that log-backtrace option doesn't give a compilation error.
//
// dont-check-compiler-stdout
// dont-check-compiler-stderr
// rustc-env:RUSTC_LOG=info
// compile-flags: -Zlog-backtrace=rustc_metadata::creader
fn main() {}
