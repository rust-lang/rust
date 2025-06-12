//@ run-pass

// This is a regression test that the metadata for the
// name_pool::methods impl in the other crate is reachable from this
// crate.

//@ aux-build:method_reexport_aux.rs

extern crate method_reexport_aux;

pub fn main() {
    use method_reexport_aux::rust::add;
    use method_reexport_aux::rust::cx;
    let x: Box<_> = Box::new(());
    x.cx();
    let y = ();
    y.add("hi".to_string());
}
