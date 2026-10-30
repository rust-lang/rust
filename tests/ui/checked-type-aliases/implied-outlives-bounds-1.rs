// Check that we expand free alias types when computing outlives-components when
// inferring outlives-bounds.

//@ revisions: pass dump
//@[pass] check-pass
#![feature(checked_type_aliases)]
#![cfg_attr(dump, feature(rustc_attrs))]

// When encountering type `&'a mut Alias<K, V>` we want to infer outlives-bounds of the form `C: 'a`
// for all outlives components `C` of `Alias<K, V>`. Contrary to other alias types, we look behind
// the free alias type when computing the components which then turn out to be `K` & `V` leading us
// to infer `K: 'a` and `V: 'a`.
#[cfg_attr(dump, rustc_dump_inferred_outlives)]
struct Type0<'a, K, V>(&'a mut Alias0<K, V>);
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE K: 'a
//[dump]~| NOTE V: 'a

type Alias0<K, V> = (K, V);

// Of course, that expansion doesn't mean that we don't consider explicit bounds on the corresp.
// type alias because we shallow walk the entire type. So below, (1) we infer `'b: 'a` & `T: 'a`
// from the `&'a ...` and the fact that the outlives-components in `Alias1<'b, T>` are `'b` & `T`.
// (2) Separately, we walk the `Alias1<'b, T>` and look at the explicit & inferred outlives-bounds
// implied by CTA `Alias1` which gives us `T: 'b`.
#[cfg_attr(dump, rustc_dump_inferred_outlives)]
struct Type1<'a, 'b, T>(&'a Alias1<'b, T>);
//[dump]~^ ERROR rustc_dump_inferred_outlives
//[dump]~| NOTE 'b: 'a
//[dump]~| NOTE T: 'a
//[dump]~| NOTE T: 'b

type Alias1<'c, U: 'c> = (&'c (), U);

fn main() {}
