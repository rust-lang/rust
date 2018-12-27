// run-pass
// aux-build:trait_inheritance_auto_xc_2_aux.rs


extern crate trait_inheritance_auto_xc_2_aux as aux;

// aux defines impls of Foo, Bar and Baz for A
use aux::{Foo, Bar, Baz, A};

// We want to extend all Foo, Bar, Bazes to Quuxes
pub trait Quux: Foo + Bar + Baz { }
impl<T:Foo + Bar + Baz> Quux for T { }

fn f<T:Quux>(a: &T) {
    assert_eq!(a.f(), 10);
    assert_eq!(a.g(), 20);
    assert_eq!(a.h(), 30);
}

pub fn main() {
    let a = &A { x: 3 };
    f(a);
}
