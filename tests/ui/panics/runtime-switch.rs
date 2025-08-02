// Test for std::panic::set_backtrace_style.

//@ compile-flags: -O
//@ compile-flags:-Cstrip=none
//@ run-fail
//@ check-run-results
//@ exec-env:RUST_BACKTRACE=0

// FIXME(#61117): Respect debuginfo-level-tests, do not force debuginfo-level=0
//@ compile-flags: -Cdebuginfo=0

// This is needed to avoid test output differences across std being built with v0 symbols vs legacy
// symbols.
//@ normalize-stderr: "begin_panic::<&str>" -> "begin_panic"
// This variant occurs on macOS with `rust.debuginfo-level = "line-tables-only"` (#133997)
//@ normalize-stderr: " begin_panic<&str>" -> " std::panicking::begin_panic"
// And this is for differences between std with and without debuginfo.
//@ normalize-stderr: "\n +at [^\n]+" -> ""

//@ ignore-msvc see #62897 and `backtrace-debuginfo.rs` test
//@ ignore-android FIXME #17520
//@ ignore-openbsd no support for libbacktrace without filename
//@ ignore-wasm no backtrace support
//@ ignore-fuchsia Backtrace not symbolized
//@ needs-subprocess

#![feature(panic_backtrace_config)]

fn main() {
    std::panic::set_backtrace_style(std::panic::BacktraceStyle::Short);
    panic!()
}
