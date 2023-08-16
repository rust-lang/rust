// gate-test-packed_bundled_libs

//@ignore-target-wasm32-unknown-unknown
//@compile-flags: -l static:+bundle,+whole-archive=rust_test_helpers --crate-type rlib
// ignore-tidy-linelength
//@error-in-other-file: link modifiers combination `+bundle,+whole-archive` is unstable when generating rlibs
// build-fail

fn main() {}
