// priv-in-pub lint tests where the private trait bounds a public type

#![crate_type = "lib"]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

#![warn(private_bounds)]

// In this test both old and new private-in-public diagnostic were emitted.
// Old diagnostic will be deleted soon.
// See https://rust-lang.github.io/rfcs/2145-type-privacy.html.

struct PrivTy;
trait PrivTr {}
pub struct PubTy;
pub struct PubTyGeneric<T>(T);
pub trait PubTr {}
impl PubTr for PrivTy {}
impl PrivTr for PubTy {}
pub trait PubTrWithAssocTy { type AssocTy; }
impl PubTrWithAssocTy for PrivTy { type AssocTy = PrivTy; }


pub struct S
//~^ ERROR private trait `PrivTr` in public interface
where
    PubTy: PrivTr
{}


pub enum E
//~^ ERROR private trait `PrivTr` in public interface
where
    PubTy: PrivTr
{}


pub fn f()
//~^ ERROR private trait `PrivTr` in public interface
where
    PubTy: PrivTr
{}


impl S
//~^ ERROR private trait `PrivTr` in public interface
where
    PubTy: PrivTr
{
    pub fn f()
    //~^ ERROR private trait `PrivTr` in public interface
    where
        PubTy: PrivTr
    {}
}


impl PubTr for PubTy
where
    PubTy: PrivTr
{}
