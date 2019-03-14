// compile-flags: --emit=metadata
// aux-build:rmeta-rlib.rs
// no-prefer-dynamic
// compile-pass

// Check that building a metadata crate works with a dependent, rlib crate.
// This is a cfail test since there is no executable to run.

extern crate rmeta_rlib;
use rmeta_rlib::Foo;

pub fn main() {
    let _ = Foo { field: 42 };
}
