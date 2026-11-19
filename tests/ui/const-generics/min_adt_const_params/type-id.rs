//@ revisions: stock gated
//@ edition:2024
//@[gated] run-pass
#![cfg_attr(gated, feature(min_adt_const_params, type_id_structural_match), allow(incomplete_features))]

use std::any::TypeId;

enum Foo {}
enum Bar {}

fn equate<const A: TypeId, const B: TypeId>() -> bool {
    //[stock]~^ ERROR `TypeId` is forbidden as the type of a const generic parameter
    //[stock]~| ERROR `TypeId` is forbidden as the type of a const generic parameter
    A == B
}

fn main() {
    assert!(equate::<{ TypeId::of::<Foo>() }, { TypeId::of::<Foo>() }>());
    assert!(equate::<{ TypeId::of::<Bar>() }, { TypeId::of::<Bar>() }>());
    assert!(!equate::<{ TypeId::of::<Foo>() }, { TypeId::of::<Bar>() }>());
}
