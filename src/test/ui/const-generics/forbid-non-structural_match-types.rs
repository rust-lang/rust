// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

#[derive(PartialEq, Eq)]
struct A;

struct B<const X: A>; // ok
//[min]~^ ERROR using `A` as const generic parameters is forbidden

struct C;

struct D<const X: C>; //~ ERROR `C` must be annotated with `#[derive(PartialEq, Eq)]`
//[min]~^ ERROR using `C` as const generic parameters is forbidden

fn main() {}
