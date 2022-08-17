#![no_std]

// @has "$.index[*][?(@.name=='Ux')].visibility" \"public\"
// @has "$.index[*][?(@.name=='Ux')].kind" \"union\"
pub union Ux {
    a: u32,
    b: u64
}

// @has "$.index[*][?(@.name=='Num')].visibility" \"public\"
// @has "$.index[*][?(@.name=='Num')].kind" \"trait\"
pub trait Num {}

// @count "$.index[*][?(@.name=='Ux')].inner.impls" 1
impl Num for Ux {}
