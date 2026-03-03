// issue: <https://github.com/rust-lang/rust/issues/13507>
// Test cross-crate TypeId stability
//@ run-pass
#![allow(unused_imports)]
//@ aux-build:typeid-cross-crate-aux.rs

extern crate typeid_cross_crate_aux;
use typeid_cross_crate_aux::testtypes;

use std::any::TypeId;

pub fn type_ids() -> Vec<TypeId> {
    use typeid_cross_crate_aux::testtypes::*;
    vec![
        TypeId::of::<FooBool>(),
        TypeId::of::<FooInt>(),
        TypeId::of::<FooUint>(),
        TypeId::of::<FooFloat>(),
        TypeId::of::<FooStr>(),
        TypeId::of::<FooArray>(),
        TypeId::of::<FooSlice>(),
        TypeId::of::<FooBox>(),
        TypeId::of::<FooPtr>(),
        TypeId::of::<FooRef>(),
        TypeId::of::<FooFnPtr>(),
        TypeId::of::<FooNil>(),
        TypeId::of::<FooTuple>(),
        TypeId::of::<dyn FooTrait>(),
        TypeId::of::<FooStruct>(),
        TypeId::of::<FooEnum>(),
    ]
}

pub fn main() {
    let othercrate = typeid_cross_crate_aux::testtypes::type_ids();
    let thiscrate = type_ids();
    assert_eq!(thiscrate, othercrate);
}
