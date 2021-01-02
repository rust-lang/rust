#![deny(unreachable_patterns)]
//~^ NOTE: lint level is defined here
pub enum TypeCtor {
    Slice,
    Array,
}

pub struct ApplicationTy(TypeCtor);

macro_rules! ty_app {
    ($ctor:pat) => {
        ApplicationTy($ctor) //~ ERROR unreachable pattern
    };
}

fn _foo(ty: ApplicationTy) {
    match ty {
        ty_app!(TypeCtor::Array) | ty_app!(TypeCtor::Slice) => {} //~ NOTE: in this expansion
    }

    // same as above, with the macro expanded
    match ty {
        ApplicationTy(TypeCtor::Array) | ApplicationTy(TypeCtor::Slice) => {}
    }
}

fn main() {}
