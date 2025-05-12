//@ proc-macro: proc-macro-panic.rs
//@ edition:2018
//@ needs-unwind proc macro panics to report errors

// Regression test for issue #76270
// Tests that we don't print an ICE message when a panic
// occurs in libproc-macro (when `-Z proc-macro-backtrace` is not specified)

extern crate proc_macro_panic;

proc_macro_panic::panic_in_libproc_macro!(); //~ ERROR proc macro panicked

fn main() {}
