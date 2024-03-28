// Check that we infer the outlives-predicates `K: 'a`, `V: 'a` for `Type`
// from the weak alias `Alias`.
// This mirrors the behavior of ADTs instead of other kinds of alias types
// like projections and opaque types.
// If we were to mirror the semantics of the latter, we would infer the
// outlives-predicate `Alias<K, V>: 'a` instead which is not what we want.

//@ revisions: default print
//@[default] check-pass

#![feature(lazy_type_alias)]
#![cfg_attr(print, feature(rustc_attrs))]
#![allow(incomplete_features)]

#[cfg_attr(print, rustc_outlives)]
struct Type<'a, K, V>(&'a mut Alias<K, V>); //[print]~ ERROR rustc_outlives

type Alias<K, V> = (K, V);

fn main() {}
