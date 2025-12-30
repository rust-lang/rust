#![feature(min_generic_const_args)]
#![expect(incomplete_features)]
// library crates exercise weirder code paths around
// DefIds which were created for const args.
#![crate_type = "lib"]

fn foo<const N: usize>() -> [(); N] {
    let a: [(); const { N }] = todo!();
    //~^ ERROR: generic parameters may not be used in const operations
}
