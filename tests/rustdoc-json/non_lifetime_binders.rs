// ignore-tidy-linelength

#![feature(non_lifetime_binders)]
#![allow(incomplete_features)]

#![no_core]
#![feature(lang_items, no_core)]

#[lang = "sized"]
pub trait Sized {}

pub trait Trait {}

#[lang = "phantom_data"]
struct PhantomData<T_>;

pub struct Wrapper<T_>(PhantomData<T_>);

// @count "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].bound_predicate.generic_params[*]" 2
// @is "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].bound_predicate.generic_params[0].name" \"\'a\"
// @is "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].bound_predicate.generic_params[0].kind" '{ "lifetime": { "outlives": [] } }'
// @is "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].bound_predicate.generic_params[1].name" \"T\"
// @is "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].bound_predicate.generic_params[1].kind" '{ "type": { "bounds": [], "default": null, "synthetic": false } }'
pub fn foo() where for<'a, T> &'a Wrapper<T>: Trait {}
