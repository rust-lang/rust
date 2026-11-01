//@ check-pass

// priv-in-pub lint tests where the private trait bounds a public type

#![crate_type = "lib"]
#![allow(incomplete_features)]

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
//~^ WARNING trait `PrivTr` is more private than the item `S`
where
    PubTy: PrivTr
{}


pub enum E
//~^ WARNING trait `PrivTr` is more private than the item `E`
where
    PubTy: PrivTr
{}


pub fn f()
//~^ WARNING trait `PrivTr` is more private than the item `f`
where
    PubTy: PrivTr
{}


impl S
//~^ WARNING trait `PrivTr` is more private than the item `S`
where
    PubTy: PrivTr
{
    pub fn f()
    //~^ WARNING trait `PrivTr` is more private than the item `S::f`
    where
        PubTy: PrivTr
    {}
}


impl PubTr for PubTy
where
    PubTy: PrivTr
{}
