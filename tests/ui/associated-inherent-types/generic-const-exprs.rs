//@ check-pass

#![feature(inherent_associated_types, generic_const_exprs)]
#![allow(incomplete_features)]

struct Parent<const O: usize>;

impl<const O: usize> Parent<O> {
    type Mapping<const I: usize> = Store<{ O + I }>
    where
        [(); O + I]:
    ;
}

struct Store<const N: usize>;

impl<const N: usize> Store<N> {
    const REIFIED: usize = N;

    fn reify() -> usize {
        N
    }
}

fn main() {
    let _ = Parent::<2>::Mapping::<{ 12 * 2 }>::REIFIED;
    let _ = Parent::<1>::Mapping::<{ 2 * 5 }>::reify();
}
