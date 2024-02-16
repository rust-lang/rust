//@ check-pass

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub trait Trait{
    type R;
    fn func(self)->Self::R;
}

pub struct TraitImpl<const N:usize>(pub i32);

impl<const N:usize> Trait for TraitImpl<N>
where [();N/2]:,
{
    type R = Self;
    fn func(self)->Self::R {
        self
    }
}

fn sample<P,Convert>(p:P,f:Convert) -> i32
where
    P:Trait,Convert:Fn(P::R)->i32
{
    f(p.func())
}

fn main() {
    let t = TraitImpl::<10>(4);
    sample(t,|x|x.0);
}
