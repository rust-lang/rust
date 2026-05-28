//@ check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]
#![crate_type = "lib"]

pub type PubAlias0 = PubTy::PrivAssocTy;
//~^ WARNING associated type `PubTy::PrivAssocTy` is more private than the item `PubAlias0`
pub type PubAlias1 = PrivTy::PubAssocTy;
//~^ WARNING type `PrivTy` is more private than the item `PubAlias1`
pub type PubAlias2 = PubTy::PubAssocTy<PrivTy>;
//~^ WARNING type `PrivTy` is more private than the item `PubAlias2`

pub struct PubTy;
impl PubTy {
    type PrivAssocTy = ();
    pub type PubAssocTy<T> = T;
}

struct PrivTy;
impl PrivTy {
    pub type PubAssocTy = ();
}
