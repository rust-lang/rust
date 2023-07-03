#![crate_type = "lib"]
#![feature(generic_const_exprs)]
#![feature(type_privacy_lints)]
#![allow(incomplete_features)]
#![warn(private_interfaces)]

// In this test both old and new private-in-public diagnostic were emitted.
// Old diagnostic will be deleted soon.
// See https://rust-lang.github.io/rfcs/2145-type-privacy.html.

pub struct Const<const U: u8>;

pub trait Trait {
    type AssocTy;
    fn assoc_fn() -> Self::AssocTy;
}

impl<const U: u8> Trait for Const<U> // OK, trait impl predicates
where
    Const<{ my_const_fn(U) }>: ,
{
    type AssocTy = Const<{ my_const_fn(U) }>;
    //~^ ERROR private type
    //~| WARNING type `fn(u8) -> u8 {my_const_fn}` is more private than the item `<Const<U> as Trait>::AssocTy`
    fn assoc_fn() -> Self::AssocTy {
        Const
    }
}

const fn my_const_fn(val: u8) -> u8 {
    // body of this function doesn't matter
    val
}
