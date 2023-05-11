// only-wasm32
// compile-flags: -C opt-level=2
// build-pass

// Regression test for #76281.
// This seems like an issue related to LLVM rather than
// libs-impl so place here.

fn main() {
    let mut v: Vec<&()> = Vec::new();
    v.sort_by_key(|&r| r as *const ());
}
