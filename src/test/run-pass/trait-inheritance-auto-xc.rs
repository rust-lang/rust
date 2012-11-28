// xfail-fast
// aux-build:trait_inheritance_auto_xc_aux.rs

extern mod aux(name = "trait_inheritance_auto_xc_aux");

use aux::{Foo, Bar, Baz, Quux};

struct A { x: int }

impl A : Foo { fn f() -> int { 10 } }
impl A : Bar { fn g() -> int { 20 } }
impl A : Baz { fn h() -> int { 30 } }

fn f<T: Quux>(a: &T) {
    assert a.f() == 10;
    assert a.g() == 20;
    assert a.h() == 30;
}

fn main() {
    let a = &A { x: 3 };
    f(a);
}

