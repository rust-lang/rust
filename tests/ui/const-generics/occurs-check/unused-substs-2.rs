#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

// The goal is to get an unevaluated const `ct` with a `Ty::Infer(TyVar(?1t)` subst.
//
// If we are then able to infer `ty::Infer(TyVar(?1t) := Ty<ct>` we introduced an
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
    //~^ ERROR mismatched types
    //~| NOTE cyclic type

    // `t` is `ty::Infer(TyVar(?1t))`
    // `foo` contains `ty::Infer(TyVar(?1t))` in its substs
    t = foo;
}
