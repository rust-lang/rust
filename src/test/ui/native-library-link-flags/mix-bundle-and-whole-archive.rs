// Mixing +bundle and +whole-archive is now allowed

// build-pass
// compile-flags: --crate-type rlib -l static:+bundle,+whole-archive=mylib
// aux-build:mylib.rs

fn main() {}
