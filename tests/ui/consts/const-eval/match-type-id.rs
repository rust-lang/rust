//@ revisions: stock gated_cmp gated_bounds gated
//@ edition:2024
#![cfg_attr(any(gated, gated_bounds), feature(type_id_structural_match))]
#![cfg_attr(any(gated, gated_cmp), feature(const_trait_impl, const_cmp))]
//@[gated] run-pass

use std::any::TypeId;

fn main() {
    const A: TypeId = TypeId::of::<u8>();

    let s: &[u8] = &[0; {
        match TypeId::of::<u8>() {
            A => 1,
            //[stock,gated_cmp]~^ ERROR constant of non-structural type `TypeId` in a pattern
            //[gated_bounds]~^^ ERROR cannot match on `TypeId` in constants
            //[gated_bounds]~| ERROR `PartialEq` is not yet stable as a const trait
            _ => panic!(),
        }
    }];

    assert_eq!(s.len(), 1);
}
