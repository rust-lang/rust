use std::fmt::Debug;

pub trait StaticOnly: 'static {}
impl<T: 'static> StaticOnly for T {}

//@ has "$.index[?(@.name=='returns_static')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
//@ has "$.index[?(@.name=='returns_static')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ has "$.index[?(@.name=='returns_static')].inner.function.sig.output.impl_trait.implied_bounds[?(@.outlives==\"'static\")]"
//@ !has "$.index[?(@.name=='returns_static')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='StaticOnly')]"
pub fn returns_static() -> impl StaticOnly {
    0u8
}

//@ has "$.index[?(@.name=='returns_maybe_unsized')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='Clone')]"
//@ has "$.index[?(@.name=='returns_maybe_unsized')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='returns_maybe_unsized')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='returns_maybe_unsized')].inner.function.sig.output.impl_trait.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
pub fn returns_maybe_unsized() -> impl Clone + ?Sized {
    123u8
}

//@ has "$.index[?(@.name=='returns_maybe_unsized_ref')].inner.function.sig.output.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has "$.index[?(@.name=='returns_maybe_unsized_ref')].inner.function.sig.output.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='returns_maybe_unsized_ref')].inner.function.sig.output.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='returns_maybe_unsized_ref')].inner.function.sig.output.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Debug')]"
pub fn returns_maybe_unsized_ref() -> &'static (impl Debug + ?Sized) {
    "hello world"
}
