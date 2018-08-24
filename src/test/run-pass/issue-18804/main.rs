// Test for issue #18804, #[linkage] does not propagate thorugh generic
// functions. Failure results in a linker error.

// ignore-asmjs no weak symbol support
// ignore-emscripten no weak symbol support

// aux-build:lib.rs

extern crate lib;

fn main() {
    lib::foo::<i32>();
}
