// ignore-tidy-linelength

#![feature(non_lifetime_binders)]
#![allow(incomplete_features)]

#![no_core]
#![feature(lang_items, no_core)]

#[lang = "sized"]
pub trait Sized {}

pub trait Trait {}

pub struct Wrapper<T_, const N_: usize>([T_; N_]);

// @count "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].bound_predicate.generic_params[*]" 3
// @is "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].bound_predicate.generic_params[0].name" \"\'a\"
// @is "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].bound_predicate.generic_params[0].kind" '{ "lifetime": { "outlives": [] } }'
// @is "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].bound_predicate.generic_params[1].name" \"T\"
// @is "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].bound_predicate.generic_params[1].kind" '{ "type": { "bounds": [], "default": null, "synthetic": false } }'
// @is "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].bound_predicate.generic_params[2].name" \"N\"
// @is "$.index[*][?(@.name=='foo')].inner.generics.where_predicates[0].bound_predicate.generic_params[2].kind" '{ "const": { "type": { "kind": "primitive", "inner": "usize" }, "default": null } }'
pub fn foo() where for<'a, T, const N: usize> &'a Wrapper<T, N>: Trait {}
