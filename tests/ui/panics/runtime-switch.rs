// Test for std::panic::set_backtrace_style.

//@ compile-flags: -O
//@ compile-flags:-Cstrip=none
//@ run-fail
//@ check-run-results
//@ exec-env:RUST_BACKTRACE=0

// This is needed to avoid test output differences across std being built with v0 symbols vs legacy
// symbols.
//@ normalize-stderr-test: "begin_panic::<&str>" -> "begin_panic"
// And this is for differences between std with and without debuginfo.
//@ normalize-stderr-test: "\n +at [^\n]+" -> ""

//@ ignore-msvc see #62897 and `backtrace-debuginfo.rs` test
//@ ignore-android FIXME #17520
//@ ignore-openbsd no support for libbacktrace without filename
//@ ignore-wasm no backtrace support
//@ ignore-emscripten no panic or subprocess support
//@ ignore-sgx no subprocess support
//@ ignore-fuchsia Backtrace not symbolized

#![feature(panic_backtrace_config)]

fn main() {
    std::panic::set_backtrace_style(std::panic::BacktraceStyle::Short);
    panic!()
}
