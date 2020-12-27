// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

#[derive(PartialEq, Eq)]
struct A;

struct B<const X: A>; // ok
//[min]~^ ERROR `A` is forbidden

struct C;

struct D<const X: C>; //~ ERROR `C` must be annotated with `#[derive(PartialEq, Eq)]`
//[min]~^ ERROR `C` is forbidden

fn main() {}
