#![feature(const_generics)]
#![allow(incomplete_features)]

// The goal is is to get an unevaluated const `ct` with a `Ty::Infer(TyVar(_#1t)` subst.
//
// If we are then able to infer `ty::Infer(TyVar(_#1t) := Ty<ct>` we introduced an
// artificial inference cycle.
struct Foo<const N: usize>;

trait Bind<T> {
    fn bind() -> (T, Self);
}

// `N` has to be `ConstKind::Unevaluated`.
impl<T> Bind<T> for Foo<{ 6 + 1 }> {
    fn bind() -> (T, Self) {
        (panic!(), Foo)
    }
}

fn main() {
    let (mut t, foo) = Foo::bind();
    // `t` is `ty::Infer(TyVar(_#1t))`
    // `foo` contains `ty::Infer(TyVar(_#1t))` in its substs
    t = foo;
    //~^ ERROR mismatched types
    //~| NOTE cyclic type
}
