// Check that we infer the outlives-predicates `K: 'a`, `V: 'a` for `Type`
// from the free alias `Alias`.
// FIXME(fmease): Proper explainer.

//@ revisions: default print
//@[default] check-pass

#![feature(lazy_type_alias)]
#![cfg_attr(print, feature(rustc_attrs))]
#![allow(incomplete_features)]

#[cfg_attr(print, rustc_outlives)]
struct Type<'a, K, V>(&'a mut Alias<K, V>); //[print]~ ERROR rustc_outlives

type Alias<K, V> = (K, V);

fn main() {}
