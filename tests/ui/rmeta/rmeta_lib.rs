// build-fail
// aux-build:rmeta-meta.rs
// no-prefer-dynamic
// error-pattern: crate `rmeta_meta` required to be available in rlib format, but was not found

// Check that building a non-metadata crate fails if a dependent crate is
// metadata-only.

extern crate rmeta_meta;
use rmeta_meta::Foo;

fn main() {
    let _ = Foo { field: 42 };
}
