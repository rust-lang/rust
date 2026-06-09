//@ build-pass

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub trait TraitWithConst {
    const SOME_CONST: usize;
}

pub trait OtherTrait: TraitWithConst {
    fn some_fn(self) -> [u8 ; <Self as TraitWithConst>::SOME_CONST];
}

impl TraitWithConst for f32 {
    const SOME_CONST: usize = 32;
}

impl OtherTrait for f32 {
    fn some_fn(self) -> [u8 ; <Self as TraitWithConst>::SOME_CONST] {
        [0; 32]
    }
}

fn main() {}
