//@ edition: 2024
use std::fmt::Debug;

pub trait StaticOnly: 'static {}
impl<T: 'static> StaticOnly for T {}

//@ has "$.index[?(@.name=='async_returns_static')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has "$.index[?(@.name=='async_returns_static')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ has "$.index[?(@.name=='async_returns_static')].inner.function.sig.output.impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
//@ !has "$.index[?(@.name=='async_returns_static')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
pub async fn async_returns_static() -> impl StaticOnly {
    0u8
}

//@ has "$.index[?(@.name=='async_returns_maybe_unsized')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has "$.index[?(@.name=='async_returns_maybe_unsized')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='async_returns_maybe_unsized')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub async fn async_returns_maybe_unsized() -> impl Debug + ?Sized {
    123
}

//@ !has "$.index[?(@.name=='async_returns_ref')].inner.function.sig.output.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='async_returns_ref')].inner.function.sig.output.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
pub async fn async_returns_ref() -> &'static (impl Debug + ?Sized) {
    "hello world"
}
