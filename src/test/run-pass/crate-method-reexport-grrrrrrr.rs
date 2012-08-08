// This is a regression test that the metadata for the
// name_pool::methods impl in the other crate is reachable from this
// crate.

// xfail-fast
// aux-build:crate-method-reexport-grrrrrrr2.rs

use crate_method_reexport_grrrrrrr2;

fn main() {
    import crate_method_reexport_grrrrrrr2::rust::add;
    import crate_method_reexport_grrrrrrr2::rust::cx;
    let x = @();
    x.cx();
    let y = ();
    y.add(~"hi");
}
