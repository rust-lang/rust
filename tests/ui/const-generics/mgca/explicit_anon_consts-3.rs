#![feature(associated_const_equality, generic_const_items, min_generic_const_args)]
#![expect(incomplete_features)]
// library crates exercise weirder code paths around
// DefIds which were created for const args.
#![crate_type = "lib"]

struct Foo<const N: usize>;

type Alias<const N: usize> = [(); const { N }];
//~^ ERROR: generic parameters may not be used in const operations
