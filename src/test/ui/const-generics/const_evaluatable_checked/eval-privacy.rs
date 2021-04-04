#![crate_type = "lib"]
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

pub struct Const<const U: u8>;

pub trait Trait {
    type AssocTy;
    fn assoc_fn() -> Self::AssocTy;
}

impl<const U: u8> Trait for Const<U>
//~^ WARN private type
//~| WARN this was previously
//~| WARN private type
//~| WARN this was previously

where
    Const<{ my_const_fn(U) }>: ,
{
    type AssocTy = Const<{ my_const_fn(U) }>;
    //~^ ERROR private type
    fn assoc_fn() -> Self::AssocTy {
        Const
    }
}

const fn my_const_fn(val: u8) -> u8 {
    // body of this function doesn't matter
    val
}
