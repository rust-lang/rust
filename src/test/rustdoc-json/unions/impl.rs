#![no_std]

// @is "$.index[*][?(@.name=='Ux')].visibility" \"public\"
// @is "$.index[*][?(@.name=='Ux')].kind" \"union\"
pub union Ux {
    a: u32,
    b: u64
}

// @is "$.index[*][?(@.name=='Num')].visibility" \"public\"
// @is "$.index[*][?(@.name=='Num')].kind" \"trait\"
pub trait Num {}

// @count "$.index[*][?(@.name=='Ux')].inner.impls" 1
impl Num for Ux {}
