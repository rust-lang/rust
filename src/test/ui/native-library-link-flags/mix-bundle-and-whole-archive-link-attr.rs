// Mixing +bundle and +whole-archive is now allowed

// build-pass
// compile-flags: --crate-type rlib
// aux-build:mylib.rs

#[link(name = "mylib", kind = "static", modifiers = "+bundle,+whole-archive")]
extern "C" {}

fn main() {}
