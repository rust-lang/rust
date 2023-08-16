// Regression test for #47429: short backtraces were not terminating correctly

//@compile-flags: -O
//@compile-flags:-Cstrip=none
// run-fail
// check-run-results
// exec-env:RUST_BACKTRACE=1

//@ignore-target-msvc see #62897 and `backtrace-debuginfo.rs` test
//@ignore-target-android FIXME #17520
//@ignore-target-openbsd no support for libbacktrace without filename
//@ignore-target-wasm no panic or subprocess support
//@ignore-target-emscripten no panic or subprocess support
//@ignore-target-sgx no subprocess support
//@ignore-target-fuchsia Backtraces not symbolized

// NOTE(eddyb) output differs between symbol mangling schemes
//@revisions: legacy v0
//@[legacy] compile-flags: -Zunstable-options -Csymbol-mangling-version=legacy
//@[v0] compile-flags: -Csymbol-mangling-version=v0

fn main() {
    panic!()
}
