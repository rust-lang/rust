#![feature(associated_const_equality, generic_const_items, min_generic_const_args)]
#![expect(incomplete_features)]
// library crates exercise weirder code paths around
// DefIds which were created for const args.
#![crate_type = "lib"]

struct Default3<const N: usize, const M: usize = const { N }>;
//~^ ERROR: generic parameters may not be used in const operations
