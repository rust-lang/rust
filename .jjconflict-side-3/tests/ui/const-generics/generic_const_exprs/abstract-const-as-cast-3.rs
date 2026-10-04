#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Trait {}
pub struct EvaluatableU128<const N: u128>;

struct HasCastInTraitImpl<const N: usize, const M: u128>;
impl<const O: usize> Trait for HasCastInTraitImpl<O, { O as u128 }> {}

pub fn use_trait_impl<const N: usize>()
where
    [(); { N + 1}]:,
    EvaluatableU128<{N as u128}>:, {
    fn assert_impl<T: Trait>() {}

    // errors are bad but seems to be pre-existing issue #86198
    assert_impl::<HasCastInTraitImpl<{ N + 1 }, { N as u128 }>>();
    //~^ ERROR mismatched types
    //~^^ ERROR unconstrained generic constant
    assert_impl::<HasCastInTraitImpl<{ N + 1 }, { N as _ }>>();
    //~^ ERROR mismatched types
    //~^^ ERROR unconstrained generic constant
    assert_impl::<HasCastInTraitImpl<13, { 12 as u128 }>>();
    //~^ ERROR mismatched types
    assert_impl::<HasCastInTraitImpl<14, 13>>();
    //~^ ERROR mismatched types
}
pub fn use_trait_impl_2<const N: usize>()
where
    [(); { N + 1}]:,
    EvaluatableU128<{N as _}>:, {
    fn assert_impl<T: Trait>() {}

    // errors are bad but seems to be pre-existing issue #86198
    assert_impl::<HasCastInTraitImpl<{ N + 1 }, { N as u128 }>>();
    //~^ ERROR mismatched types
    //~^^ ERROR unconstrained generic constant
    assert_impl::<HasCastInTraitImpl<{ N + 1 }, { N as _ }>>();
    //~^ ERROR mismatched types
    //~^^ ERROR unconstrained generic constant
    assert_impl::<HasCastInTraitImpl<13, { 12 as u128 }>>();
    //~^ ERROR mismatched types
    assert_impl::<HasCastInTraitImpl<14, 13>>();
    //~^ ERROR mismatched types
}

fn main() {}
