// Verify that a type is considered reachable when its constructor is
// reachable. The auxiliary library is constructed so that all types are
// shadowed and cannot be named directly, while their constructors are
// reexported. Regression test for issue #96934.
//
//@ aux-build:ctor_aux.rs
//@ edition:2021
//@ build-pass

extern crate ctor_aux;

fn main() {
    ctor_aux::s.f();
    ctor_aux::x.g();
    ctor_aux::y.g();
}
