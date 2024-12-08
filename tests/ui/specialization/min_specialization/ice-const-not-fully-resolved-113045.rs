// ICE min_specialization:
// Ok(['?0, Const { ty: usize, kind: Leaf(0x0000000000000000) }]) is not fully resolved
// issue: rust-lang/rust#113045

#![feature(min_specialization)]

trait X {}

impl<'a, const N: usize> X for [(); N] {}

impl<'a, Unconstrained> X for [(); 0] {}
//~^ ERROR the type parameter `Unconstrained` is not constrained by the impl trait, self type, or predicates
//~| ERROR specialization impl does not specialize any associated items

fn main() {}
