#![feature(associated_const_equality, generic_const_items, min_generic_const_args)]
#![expect(incomplete_features)]
// library crates exercise weirder code paths around
// DefIds which were created for const args.
#![crate_type = "lib"]

trait Trait {
    #[type_const]
    const ASSOC: usize;
}

fn ace_bounds<
    const N: usize,
    T: Trait<ASSOC = const { N }>,
    //~^ ERROR: generic parameters may not be used in const operations
>() {}
