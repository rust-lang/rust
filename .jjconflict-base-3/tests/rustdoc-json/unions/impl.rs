#![no_std]

//@ is "$.index[?(@.name=='Ux')].visibility" \"public\"
//@ has "$.index[?(@.name=='Ux')].inner.union"
pub union Ux {
    a: u32,
    b: u64,
}

//@ is "$.index[?(@.name=='Num')].visibility" \"public\"
//@ has "$.index[?(@.name=='Num')].inner.trait"
pub trait Num {}

//@ count "$.index[?(@.name=='Ux')].inner.union.impls" 1
impl Num for Ux {}
