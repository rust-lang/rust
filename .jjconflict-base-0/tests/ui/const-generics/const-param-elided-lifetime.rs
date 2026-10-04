// Elided lifetimes within the type of a const generic parameters is disallowed. This matches the
// behaviour of trait bounds where `fn foo<T: Ord<&u8>>() {}` is illegal. Though we could change
// elided lifetimes within the type of a const generic parameters to be 'static, like elided
// lifetimes within const/static items.
//@ revisions: full min
#![cfg_attr(full, feature(adt_const_params, unsized_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

struct A<const N: &u8>;
//~^ ERROR `&` without an explicit lifetime name cannot be used here
trait B {}

impl<const N: &u8> A<N> {
    //~^ ERROR `&` without an explicit lifetime name cannot be used here
    fn foo<const M: &u8>(&self) {}
    //~^ ERROR `&` without an explicit lifetime name cannot be used here
}

impl<const N: &u8> B for A<N> {}
//~^ ERROR `&` without an explicit lifetime name cannot be used here

fn bar<const N: &u8>() {}
//~^ ERROR `&` without an explicit lifetime name cannot be used here

fn main() {}
