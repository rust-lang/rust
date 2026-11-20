#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub trait LibTrait {
    const NUM: usize;
    fn configure(&mut self, cfg: &Config<{ Self::NUM }>);
}

pub struct Config<const N: usize> {
    pub config: [u8; N],
}

pub struct ImplementsTraitOverConstGeneric<const N: usize>;

impl<const N: usize> LibTrait for ImplementsTraitOverConstGeneric<N> {
    const NUM: usize = N;
    fn configure(&mut self, _: &Config<{ Self::NUM }>) {}
}
