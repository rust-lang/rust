#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub trait Trait {}
struct HasCastInTraitImpl<const N: usize, const M: usize>;
impl<const M: usize> Trait for HasCastInTraitImpl<M, { M + 1 }> {}
pub struct HasTrait<T: Trait>(T);

fn foo1<const N: usize>() -> HasTrait<HasCastInTraitImpl<{ N + 2}, { N }>> {
    //~^ ERROR mismatched types
    //~| ERROR unconstrained generic constant
    loop {}
}

fn foo2<const N: usize>() -> HasTrait<HasCastInTraitImpl<{ N + 1}, { N + 1 }>> {
    //~^ ERROR mismatched types
    //~| ERROR unconstrained generic constant
    loop {}
}

fn foo3<const N: usize>() -> HasTrait<HasCastInTraitImpl<{ N + 1}, { N - 1}>> {
    //~^ ERROR mismatched types
    //~| ERROR unconstrained generic constant
    loop {}
}

fn foo4<const N: usize>(c : [usize; N]) -> HasTrait<HasCastInTraitImpl<{ N - 1}, { N }>> {
    //~^ ERROR mismatched types
    //~| ERROR unconstrained generic constant
    loop {}
}

fn foo5<const N: usize>() -> HasTrait<HasCastInTraitImpl<{ N + N }, { 2 * N }>> {
    //~^ ERROR mismatched types
    //~| ERROR unconstrained generic constant
    loop {}
}

fn main() {}
