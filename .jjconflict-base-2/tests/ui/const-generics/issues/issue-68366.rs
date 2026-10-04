// Checks that const expressions have a useful note explaining why they can't be evaluated.
// The note should relate to the fact that it cannot be shown forall N that it maps 1-1 to a new
// type.

//@ revisions: full min
#![cfg_attr(full, feature(generic_const_exprs))]
#![cfg_attr(full, allow(incomplete_features))]

struct Collatz<const N: Option<usize>>;
//~^ ERROR: `Option<usize>` is forbidden

impl <const N: usize> Collatz<{Some(N)}> {}
//~^ ERROR the const parameter
//[min]~^^ ERROR generic parameters may not be used in const operations
//[full]~^^^ ERROR overly complex

struct Foo;

impl<const N: usize> Foo {}
//~^ ERROR the const parameter

fn main() {}
