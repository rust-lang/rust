#![feature(const_generics)]
#![allow(incomplete_features)]

// The goal is is to get an unevaluated const `ct` with a `Ty::Infer(TyVar(_#1t)` subst.
//
// If we are then able to infer `ty::Infer(TyVar(_#1t) := Ty<ct>` we introduced an
// artificial inference cycle.
fn bind<T>() -> (T, [u8; 6 + 1]) {
    todo!()
}

fn main() {
    let (mut t, foo) = bind();
    // `t` is `ty::Infer(TyVar(_#1t))`
    // `foo` contains `ty::Infer(TyVar(_#1t))` in its substs
    t = foo;
    //~^ ERROR mismatched types
    //~| NOTE cyclic type
}
