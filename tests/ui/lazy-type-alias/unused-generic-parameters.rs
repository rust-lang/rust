// Check that we reject bivariant generic parameters as unused.
// Furthermore, check that we only emit a single diagnostic for unused type parameters:
// Previously, we would emit *two* errors, namely E0392 and E0091.

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type A<'a> = ();
//~^ ERROR lifetime parameter `'a` is never used
//~| HELP consider removing `'a`

type B<T> = ();
//~^ ERROR type parameter `T` is never used
//~| HELP consider removing `T`
//~| HELP if you intended `T` to be a const parameter

// Check that we don't emit the const param help message here:
type C<T: Copy> = ();
//~^ ERROR type parameter `T` is never used
//~| HELP consider removing `T`

fn main() {}
