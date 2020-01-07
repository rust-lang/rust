// build-pass
// compile-flags: --crate-type lib

// Regression test for LLVM crash affecting Emscripten targets

pub fn foo() {
    (0..0).rev().next();
}
