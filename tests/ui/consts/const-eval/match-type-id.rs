//@ revisions: stock gated
#![cfg_attr(gated, feature(const_trait_impl, const_cmp))]
//@[gated] run-pass

use std::any::TypeId;

fn main() {
    const A: TypeId = TypeId::of::<u8>();

    let s: &[u8] = &[0; {
        match TypeId::of::<u8>() {
            A => 1,
            //[stock]~^ ERROR cannot match on `TypeId` in constants
            //[stock]~| ERROR `PartialEq` is not yet stable as a const trait
            _ => panic!(),
        }
    }];

    assert_eq!(s.len(), 1);
}
