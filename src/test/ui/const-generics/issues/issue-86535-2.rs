// run-pass
#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

pub trait Foo {
    const ASSOC_C: usize;
    fn foo() where [(); Self::ASSOC_C]:;
}

struct Bar<const N: &'static ()>;
impl<const N: &'static ()> Foo for Bar<N> {
    const ASSOC_C: usize = 3;

    fn foo() where [u8; Self::ASSOC_C]: {
        let _: [u8; Self::ASSOC_C] = loop {};
    }
}

fn main() {}
