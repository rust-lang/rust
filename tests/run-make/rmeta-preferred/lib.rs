// Test that using rlibs and rmeta dep crates work together. Specifically, that
// there can be both an rmeta and an rlib file and rustc will prefer the rmeta
// file.
//
// This behavior is simply making sure this doesn't accidentally change; in this
// case we want to make sure that the rlib isn't being used as that would cause
// bugs in -Zbinary-dep-depinfo (see #68298).

extern crate rmeta_aux;
use rmeta_aux::Foo;

pub fn foo() {
    let _ = Foo { field: 42 };
}
