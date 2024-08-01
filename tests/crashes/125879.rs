//@ known-bug: rust-lang/rust#125879
#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

pub type PubAlias0 = PubTy::PrivAssocTy;

pub struct PubTy;
impl PubTy {
    type PrivAssocTy = ();
}

pub struct S(pub PubAlias0);

pub unsafe fn foo(a: S) -> S {
    a
}

fn main() {}
