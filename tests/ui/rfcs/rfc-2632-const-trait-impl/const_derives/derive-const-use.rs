// known-bug: #110395
#![feature(const_trait_impl, const_cmp, const_default_impls, derive_const)]

pub struct A;

impl const Default for A {
    fn default() -> A { A }
}

impl const PartialEq for A {
    fn eq(&self, _: &A) -> bool { true }
}

#[derive_const(Default, PartialEq)]
pub struct S((), A);

const _: () = assert!(S((), A) == S::default());

fn main() {}
