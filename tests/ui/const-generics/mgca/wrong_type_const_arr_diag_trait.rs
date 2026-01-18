// This test causes ERROR: mismatched types [E0308]
// and makes rustc to print array from const arguments
#![feature(min_generic_const_args, adt_const_params, trivial_bounds)]
#![allow(incomplete_features)]

trait Trait {
    #[type_const]
    const ASSOC: u8;
}

struct TakesArr<const N: [u8; 1]>;

fn foo<const N: u8>()
where
    u8: Trait
{
    let _: TakesArr<{ [<u8 as Trait>::ASSOC] }> = TakesArr::<{ [1] }>;
    //~^ ERROR: mismatched types [E0308]
}

fn main() {}
