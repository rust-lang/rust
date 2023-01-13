// Test that the "C-unwind" ABI is feature-gated, and *can* be used when the
// `c_unwind` feature gate is enabled.

// check-pass

#![feature(c_unwind)]

extern "C-unwind" fn f() {}

fn main() {
    f();
}
