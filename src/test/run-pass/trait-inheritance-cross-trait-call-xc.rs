// xfail-fast
// aux-build:trait_inheritance_cross_trait_call_xc_aux.rs

extern mod aux(name = "trait_inheritance_cross_trait_call_xc_aux");

trait Bar : aux::Foo {
    fn g() -> int;
}

impl aux::A : Bar {
    fn g() -> int { self.f() }
}

fn main() {
    let a = &aux::A { x: 3 };
    assert a.g() == 10;
}

