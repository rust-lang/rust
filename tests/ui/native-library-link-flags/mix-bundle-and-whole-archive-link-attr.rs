// gate-test-packed_bundled_libs

//@ignore-target-wasm32-unknown-unknown
//@compile-flags: --crate-type rlib
// ignore-tidy-linelength
//@error-in-other-file: link modifiers combination `+bundle,+whole-archive` is unstable when generating rlibs
// build-fail

#[link(name = "rust_test_helpers", kind = "static", modifiers = "+bundle,+whole-archive")]
extern "C" {}

fn main() {}
