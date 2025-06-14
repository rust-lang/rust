//@ check-pass

#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

pub type PubAlias0 = PubTy::PrivAssocTy;
//~^ WARN: associated type `PubTy::PrivAssocTy` is more private than the item `PubAlias0`

pub struct PubTy;
impl PubTy {
    type PrivAssocTy = ();
}

pub struct S(pub PubAlias0);
//~^ WARN: associated type `PubTy::PrivAssocTy` is more private than the item `S::0`

pub unsafe fn foo(a: S) -> S {
    a
}

fn main() {}
