// aux-build:const_generic_lib.rs
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

extern crate const_generic_lib;

fn main() {
    let _ = const_generic_lib::function(const_generic_lib::Struct([0u8, 1u8]));
    //~^ ERROR mismatched types
    let _: const_generic_lib::Alias = const_generic_lib::Struct([0u8, 1u8, 2u8]);
    //~^ ERROR mismatched types
}
