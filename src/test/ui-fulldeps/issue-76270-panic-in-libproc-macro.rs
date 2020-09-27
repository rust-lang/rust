// aux-build:proc-macro-panic.rs
// edition:2018
// ignore-stage1
// only-linux
//
// FIXME: This should be a normal (stage1, all platforms) test in
// src/test/ui/proc-macro once issue #59998 is fixed.

// Regression test for issue #76270
// Tests that we don't print an ICE message when a panic
// occurs in libproc-macro (when `-Z proc-macro-backtrace` is not specified)

extern crate proc_macro_panic;

proc_macro_panic::panic_in_libproc_macro!(); //~ ERROR proc macro panicked

fn main() {}
