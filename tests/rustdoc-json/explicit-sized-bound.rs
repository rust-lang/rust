//! When a `Sized` bound is explicitly given, it appears in rustdoc JSON too.

//@ has "$.index[?(@.name=='explicitly_sized_generic')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='explicitly_sized_generic')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.modifier=='maybe')]"
pub fn explicitly_sized_generic<T: Sized>(value: T) -> T {
    value
}

//@ has "$.index[?(@.name=='explicitly_sized_generic_where_clause')].inner.function.generics.where_predicates[0].bound_predicate.bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='explicitly_sized_generic_where_clause')].inner.function.generics.where_predicates[0].bound_predicate.bounds[?(@.trait_bound.modifier=='maybe')]"
pub fn explicitly_sized_generic_where_clause<T>(value: T) -> T
where
    T: Sized,
{
    value
}

//@ has "$.index[?(@.name=='explicitly_sized_impl_trait')].inner.function.sig.inputs[0][1].impl_trait.bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='explicitly_sized_impl_trait')].inner.function.sig.inputs[0][1].impl_trait.bounds[?(@.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='explicitly_sized_impl_trait')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='explicitly_sized_impl_trait')].inner.function.sig.output.impl_trait.bounds[?(@.trait_bound.modifier=='maybe')]"
pub fn explicitly_sized_impl_trait(value: impl Sized) -> impl Sized {
    value
}

pub trait Example {
    //@ has "$.index[?(@.name=='Explicit')].inner.assoc_type.bounds[?(@.trait_bound.trait.path=='Sized')]"
    //@ !has "$.index[?(@.name=='Explicit')].inner.assoc_type.bounds[?(@.trait_bound.modifier=='maybe')]"
    type Explicit: Sized;
}
