//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Trait {}
pub struct EvaluatableU128<const N: u128>;

struct HasCastInTraitImpl<const N: usize, const M: u128>;
impl<const O: usize> Trait for HasCastInTraitImpl<O, { O as u128 }> {}

pub fn use_trait_impl<const N: usize>() where EvaluatableU128<{N as u128}>:, {
    fn assert_impl<T: Trait>() {}

    assert_impl::<HasCastInTraitImpl<N, { N as u128 }>>();
    assert_impl::<HasCastInTraitImpl<N, { N as _ }>>();
    assert_impl::<HasCastInTraitImpl<12, { 12 as u128 }>>();
    assert_impl::<HasCastInTraitImpl<13, 13>>();
}
pub fn use_trait_impl_2<const N: usize>() where EvaluatableU128<{N as _}>:, {
    fn assert_impl<T: Trait>() {}

    assert_impl::<HasCastInTraitImpl<N, { N as u128 }>>();
    assert_impl::<HasCastInTraitImpl<N, { N as _ }>>();
    assert_impl::<HasCastInTraitImpl<12, { 12 as u128 }>>();
    assert_impl::<HasCastInTraitImpl<13, 13>>();
}


fn main() {}
