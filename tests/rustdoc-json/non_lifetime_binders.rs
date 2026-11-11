#![feature(non_lifetime_binders)]
#![allow(incomplete_features)]

pub trait Trait {}

pub struct Wrapper<T_>(std::marker::PhantomData<T_>);

//@ count "$.index[?(@.name=='foo')].inner.function.generics.where_predicates[0].bound_predicate.generic_params[*]" 2
//@ is "$.index[?(@.name=='foo')].inner.function.generics.where_predicates[0].bound_predicate.generic_params[0].name" \"\'a\"
//@ is "$.index[?(@.name=='foo')].inner.function.generics.where_predicates[0].bound_predicate.generic_params[0].kind" '{ "lifetime": { "outlives": [] } }'
//@ is "$.index[?(@.name=='foo')].inner.function.generics.where_predicates[0].bound_predicate.generic_params[1].name" \"T\"
//@ is "$.index[?(@.name=='foo')].inner.function.generics.where_predicates[0].bound_predicate.generic_params[1].kind" '{ "type": { "bounds": [], "default": null, "is_synthetic": false } }'
pub fn foo()
where
    for<'a, T> &'a Wrapper<T>: Trait,
{
}
