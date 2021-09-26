// run-pass
// pretty-expanded FIXME #23616

// This is a regression test that the metadata for the
// name_pool::methods impl in the other crate is reachable from this
// crate.

// aux-build:crate-method-reexport-grrrrrrr2.rs

extern crate crate_method_reexport_grrrrrrr2;

pub fn main() {
    use crate_method_reexport_grrrrrrr2::rust::add;
    use crate_method_reexport_grrrrrrr2::rust::cx;
    let x: Box<_> = Box::new(());
    x.cx();
    let y = ();
    y.add("hi".to_string());
}
