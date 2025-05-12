#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

// The goal is to get an unevaluated const `ct` with a `Ty::Infer(TyVar(?1t)` subst.
//
// If we are then able to infer `ty::Infer(TyVar(?1t) := Ty<ct>` we introduced an
// artificial inference cycle.
fn bind<T>() -> (T, [u8; 6 + 1]) {
    todo!()
}

fn main() {
    let (mut t, foo) = bind();
    //~^ ERROR mismatched types
    //~| NOTE cyclic type

    // `t` is `ty::Infer(TyVar(?1t))`
    // `foo` contains `ty::Infer(TyVar(?1t))` in its substs
    t = foo;

}
