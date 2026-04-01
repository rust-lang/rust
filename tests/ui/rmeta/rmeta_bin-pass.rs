//@ compile-flags: --emit=obj,metadata --crate-type=bin
//@ aux-build:rmeta-meta.rs
//@ no-prefer-dynamic
//@ build-pass

// Check that building a metadata bin crate works with a dependent, metadata
// crate if linking is not requested.

extern crate rmeta_meta;
use rmeta_meta::Foo;

pub fn main() {
    let _ = Foo { field: 42 };
}
