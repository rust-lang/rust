// run-pass
// Test for issue #18804, #[linkage] does not propagate through generic
// functions. Failure results in a linker error.

// ignore-asmjs no weak symbol support
// ignore-emscripten no weak symbol support
// ignore-windows no extern_weak linkage
// ignore-macos no extern_weak linkage

// aux-build:lib.rs

// rust-lang/rust#56772: nikic says we need this to be proper test.
// compile-flags: -C no-prepopulate-passes -C passes=name-anon-globals

extern crate lib;

fn main() {
    lib::foo::<i32>();
}
