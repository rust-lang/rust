// xfail-fast
// aux-build:trait_inheritance_auto_xc_2_aux.rs

extern mod aux(name = "trait_inheritance_auto_xc_2_aux");

// aux defines impls of Foo, Bar and Baz for A
use aux::{Foo, Bar, Baz, A};

// We want to extend all Foo, Bar, Bazes to Quuxes
pub trait Quux: Foo, Bar, Baz { }
impl<T: Foo Bar Baz> T: Quux { }

fn f<T: Quux>(a: &T) {
    assert a.f() == 10;
    assert a.g() == 20;
    assert a.h() == 30;
}

fn main() {
    let a = &A { x: 3 };
    f(a);
}

