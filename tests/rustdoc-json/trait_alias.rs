#![feature(trait_alias)]

//@ set StrLike = "$.index[?(@.name=='StrLike')].id"
//@ is "$.index[?(@.name=='StrLike')].visibility" \"public\"
//@ has "$.index[?(@.name=='StrLike')].inner.trait_alias"
//@ is "$.index[?(@.name=='StrLike')].span.filename" $FILE
pub trait StrLike = AsRef<str>;

//@ is "$.index[?(@.name=='f')].inner.function.sig.output.impl_trait[0].trait_bound.trait.id" $StrLike
pub fn f() -> impl StrLike {
    "heya"
}

//@ !is "$.index[?(@.name=='g')].inner.function.sig.output.impl_trait[0].trait_bound.trait.id" $StrLike
pub fn g() -> impl AsRef<str> {
    "heya"
}
