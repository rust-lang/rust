// This test ensures that rustdoc does not panic on inherented associated types
// that are referred to without fully-qualified syntax.

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

pub struct Struct;

impl Struct {
    pub type AssocTy = usize;
    pub const AssocConst: Self::AssocTy = 42;
    //~^ ERROR ambiguous associated type
    //~| HELP use fully-qualified syntax
    // FIXME: for some reason, the error is shown twice with rustdoc but only once with rustc
    //~| ERROR ambiguous associated type
    //~| HELP use fully-qualified syntax
}
