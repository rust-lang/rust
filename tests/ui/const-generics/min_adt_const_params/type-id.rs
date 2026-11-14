//@ edition:2024
//@run-pass
#![feature(min_adt_const_params)]
#![allow(incomplete_features)]

use std::any::TypeId;

enum Foo {}
enum Bar {}

fn equate<const A: TypeId, const B: TypeId>() -> bool {
    A == B
}

fn main() {
    assert!(equate::<{ TypeId::of::<Foo>() }, { TypeId::of::<Foo>() }>());
    assert!(equate::<{ TypeId::of::<Bar>() }, { TypeId::of::<Bar>() }>());
    assert!(!equate::<{ TypeId::of::<Foo>() }, { TypeId::of::<Bar>() }>());
}
