// run-pass
// aux-build:issue-12133-rlib.rs
// aux-build:issue-12133-dylib.rs
// aux-build:issue-12133-dylib2.rs
// ignore-emscripten no dylib support
// ignore-musl
// needs-dynamic-linking

// pretty-expanded FIXME #23616

extern crate issue_12133_dylib2 as other;

fn main() {}
