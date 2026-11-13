// priv-in-pub lint tests where the private type appears in the
// `where` clause of a public item

#![crate_type = "lib"]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct PrivTy;
trait PrivTr {}
pub struct PubTy;
pub struct PubTyGeneric<T>(T);
pub trait PubTr {}
impl PubTr for PrivTy {}
pub trait PubTrWithAssocTy {
    type AssocTy;
}
impl PubTrWithAssocTy for PrivTy {
    type AssocTy = PrivTy;
}

pub struct S
//~^ WARNING type `PrivTy` is more private than the item `S`
where
    PrivTy:, {}

pub enum E
//~^ WARNING type `PrivTy` is more private than the item `E`
where
    PrivTy:, {}

pub fn f()
//~^ WARNING type `PrivTy` is more private than the item `f`
where
    PrivTy:,
{
}

impl S
//~^ WARNING type `PrivTy` is more private than the item `S`
where
    PrivTy:,
{
    pub fn f()
    //~^ WARNING type `PrivTy` is more private than the item `S::f`
    where
        PrivTy:,
    {
    }
}

impl PubTr for PubTy where PrivTy: {}

impl<T> PubTr for PubTyGeneric<T> where T: PubTrWithAssocTy<AssocTy = PrivTy> {}

pub struct Const<const U: u8>;

pub trait Trait {
    type AssocTy;
    fn assoc_fn() -> Self::AssocTy;
}

impl<const U: u8> Trait for Const<U>
where
    Const<{ my_const_fn(U) }>:,
{
    type AssocTy = Const<{ my_const_fn(U) }>;
    //~^ ERROR private type
    //~| ERROR private type
    fn assoc_fn() -> Self::AssocTy {
        Const
    }
}

const fn my_const_fn(val: u8) -> u8 {
    // body of this function doesn't matter
    val
}
