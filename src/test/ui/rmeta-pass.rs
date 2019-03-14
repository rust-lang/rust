// compile-flags: --emit=metadata
// aux-build:rmeta-meta.rs
// no-prefer-dynamic
// compile-pass

// Check that building a metadata crate works with a dependent, metadata-only
// crate.
// This is a cfail test since there is no executable to run.

extern crate rmeta_meta;
use rmeta_meta::Foo;

pub fn main() {
    let _ = Foo { field: 42 };
}
