#![feature(trait_alias)]

//@ set StrLike = "$.index[?(@.name=='StrLike')].id"
//@ is "$.index[?(@.name=='StrLike')].visibility" \"public\"
//@ has "$.index[?(@.name=='StrLike')].inner.trait_alias"
//@ is "$.index[?(@.name=='StrLike')].span.filename" $FILE
pub trait StrLike = AsRef<str>;

//@ is "$.index[?(@.name=='f')].inner.function.sig.output" 1
//@ is "$.types[1].impl_trait[0].trait_bound.trait.id" $StrLike
pub fn f() -> impl StrLike {
    "heya"
}

//@ is "$.index[?(@.name=='g')].inner.function.sig.output" 2
//@ !is "$.types[2].impl_trait[0].trait_bound.trait.id" $StrLike
pub fn g() -> impl AsRef<str> {
    "heya"
}
