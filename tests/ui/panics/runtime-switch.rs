// Test for std::panic::set_backtrace_style.

//@ compile-flags: -O
//@ compile-flags:-Cstrip=none
//@ run-fail
//@ check-run-results
//@ exec-env:RUST_BACKTRACE=0

//@ normalize-stderr: "omitted [0-9]+ frames?" -> "omitted N frames"
//@ normalize-stderr: ".rs:[0-9]+:[0-9]+" -> ".rs:LL:CC"

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
