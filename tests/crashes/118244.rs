//@ known-bug: #118244
//@ compile-flags: -Cdebuginfo=2

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
struct Inner<const N: usize, const M: usize>;
impl<const N: usize, const M: usize> Inner<N, M> where [(); N + M]: {
    fn i() -> Self {
        Self
    }
}

struct Outer<const A: usize, const B: usize>(Inner<A, { B * 2 }>) where [(); A + (B * 2)]:;
impl<const A: usize, const B: usize> Outer<A, B> where [(); A + (B * 2)]: {
    fn o() -> Self {
        Self(Inner::i())
    }
}

fn main() {
    Outer::<1, 1>::o();
}
