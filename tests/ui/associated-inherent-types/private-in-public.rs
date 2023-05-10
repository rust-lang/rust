#![feature(inherent_associated_types)]
#![allow(incomplete_features)]
#![crate_type = "lib"]

#![deny(private_in_public)]

pub type PubAlias0 = PubTy::PrivAssocTy;
//~^ ERROR private associated type `PubTy::PrivAssocTy` in public interface (error E0446)
//~| WARNING this was previously accepted
pub type PubAlias1 = PrivTy::PubAssocTy;
//~^ ERROR private type `PrivTy` in public interface (error E0446)
//~| WARNING this was previously accepted
pub type PubAlias2 = PubTy::PubAssocTy<PrivTy>;
//~^ ERROR private type `PrivTy` in public interface (error E0446)
//~| WARNING this was previously accepted

pub struct PubTy;
impl PubTy {
    type PrivAssocTy = ();
    pub type PubAssocTy<T> = T;
}

struct PrivTy;
impl PrivTy {
    pub type PubAssocTy = ();
}
