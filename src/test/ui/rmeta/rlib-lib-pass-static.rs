// aux-build:rmeta-mir-static.rs
// no-prefer-dynamic
// build-pass (FIXME(62277): could be check-pass?)

// Check that building a an rlib crate dependent on a rmeta crate can
// use statics, consts and type aliases.

#![crate_type="rlib"]

extern crate rmeta_mir_static;
use rmeta_mir_static::{FOO, BAR};

pub fn main() {
    println!("foo {} bar {}", FOO, BAR);
}
