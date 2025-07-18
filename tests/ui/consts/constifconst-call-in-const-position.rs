//@ known-bug: #102498

#![feature(const_trait_impl, generic_const_exprs)]
#![allow(incomplete_features)]

#[const_trait]
pub trait Tr {
    fn a() -> usize;
}

impl Tr for () {
    fn a() -> usize {
        1
    }
}

const fn foo<T: [const] Tr>() -> [u8; T::a()] {
    [0; T::a()]
}

fn main() {
    foo::<()>();
}
