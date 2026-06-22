//@ check-pass
#![feature(const_trait_impl, const_default, const_cmp, derive_const)]

pub struct A;

const impl Default for A {
    fn default() -> A {
        A
    }
}

const impl PartialEq for A {
    fn eq(&self, _: &A) -> bool {
        true
    }
}

#[derive_const(Default, PartialEq)]
pub struct S((), A);

const _: () = assert!(S((), A) == S::default());

fn main() {}
