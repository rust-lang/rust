#![feature(unstable_test_feature)]

//@ aux-build:unstable.rs

extern crate unstable;

use unstable::UnstableStruct;

fn main() {
    let UnstableStruct { stable, stable2, } = UnstableStruct::default();
    //~^ ERROR pattern does not mention field `unstable`

    let UnstableStruct { stable, unstable, } = UnstableStruct::default();
    //~^ ERROR pattern does not mention field `stable2`

    // OK: stable field is matched
    let UnstableStruct { stable, stable2, unstable } = UnstableStruct::default();
}
