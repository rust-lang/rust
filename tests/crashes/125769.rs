//@ known-bug: rust-lang/rust#125769

#![feature(generic_const_exprs)]

trait Trait {}

struct HasCastInTraitImpl<const N: usize, const M: u128>;
impl<const O: f64> Trait for HasCastInTraitImpl<O, { O as u128 }> {}

pub fn use_trait_impl() {
    fn assert_impl<T: Trait>() {}

    assert_impl::<HasCastInTraitImpl<13, 13>>();
}
