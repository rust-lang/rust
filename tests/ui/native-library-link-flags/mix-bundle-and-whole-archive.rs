// gate-test-packed_bundled_libs

// ignore-wasm32-bare
// compile-flags: -l static:+bundle,+whole-archive=rust_test_helpers --crate-type rlib
// error-pattern: link modifiers combination `+bundle,+whole-archive` is unstable when generating rlibs
// build-fail

fn main() {}
