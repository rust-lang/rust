#![feature(inherent_associated_types)]
#![allow(incomplete_features)]
#![crate_type = "lib"]

#![deny(private_in_public)]
#![warn(private_interfaces)]

// In this test both old and new private-in-public diagnostic were emitted.
// Old diagnostic will be deleted soon.
// See https://rust-lang.github.io/rfcs/2145-type-privacy.html.

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
