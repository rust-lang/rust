// Test that using rlibs and rmeta dep crates work together. Specifically, that
// there can be both an rmeta and an rlib file and rustc will prefer the rlib.

// aux-build:rmeta_rmeta.rs
// aux-build:rmeta_rlib.rs

extern crate rmeta_aux;
use rmeta_aux::Foo;

pub fn main() {
    let _ = Foo { field: 42 };
}
