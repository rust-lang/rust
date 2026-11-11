//@ check-pass
#![deny(unreachable_patterns)]
pub enum TypeCtor {
    Slice,
    Array,
}

pub struct ApplicationTy(TypeCtor);

macro_rules! ty_app {
    ($ctor:pat) => {
        ApplicationTy($ctor)
    };
}

fn _foo(ty: ApplicationTy) {
    match ty {
        ty_app!(TypeCtor::Array) | ty_app!(TypeCtor::Slice) => {}
    }

    // same as above, with the macro expanded
    match ty {
        ApplicationTy(TypeCtor::Array) | ApplicationTy(TypeCtor::Slice) => {}
    }
}

fn main() {}
