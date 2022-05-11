// Elided lifetimes within the type of a const generic parameters is disallowed. This matches the
// behaviour of trait bounds where `fn foo<T: Ord<&u8>>() {}` is illegal. Though we could change
// elided lifetimes within the type of a const generic parameters to be 'static, like elided
// lifetimes within const/static items.
// revisions: full min
#![cfg_attr(full, feature(adt_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

struct A<const N: &u8>;
//~^ ERROR missing lifetime specifier [E0106]

trait B {}

impl<const N: &u8> A<N> {
//~^ ERROR missing lifetime specifier [E0106]
    fn foo<const M: &u8>(&self) {}
    //~^ ERROR missing lifetime specifier [E0106]
}

impl<const N: &u8> B for A<N> {}
//~^ ERROR missing lifetime specifier [E0106]

fn bar<const N: &u8>() {}
//~^ ERROR missing lifetime specifier [E0106]

fn main() {}
