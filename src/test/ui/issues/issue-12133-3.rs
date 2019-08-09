// run-pass
// aux-build:issue-12133-rlib.rs
// aux-build:issue-12133-dylib.rs
// aux-build:issue-12133-dylib2.rs
// ignore-cloudabi no dylib support
// ignore-emscripten no dylib support
// ignore-musl
// ignore-sgx no dylib support

// pretty-expanded FIXME #23616

extern crate issue_12133_dylib2 as other;

fn main() {}
