// run-pass
//
// This test makes sure that log-backtrace option at least parses correctly
//
// dont-check-compiler-stdout
// dont-check-compiler-stderr
// rustc-env:RUSTC_LOG=info
// rustc-env:RUSTC_LOG_BACKTRACE=rustc_metadata::creader
fn main() {}
