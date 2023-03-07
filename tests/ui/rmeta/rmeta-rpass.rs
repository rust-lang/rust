// run-pass
// Test that using rlibs and rmeta dep crates work together. Specifically, that
// there can be both an rmeta and an rlib file and rustc will prefer the rmeta
// file.
//
// This behavior is simply making sure this doesn't accidentally change; in this
// case we want to make sure that the rlib isn't being used as that would cause
// bugs in -Zbinary-dep-depinfo (see #68298).

// aux-build:rmeta-rmeta.rs
// aux-build:rmeta-rlib-rpass.rs

extern crate rmeta_aux;
use rmeta_aux::Foo;

pub fn main() {
    let _ = Foo { field2: 42 };
}
