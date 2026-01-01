//@ compile-flags: --crate-type=lib
#![allow(internal_features)]
#![feature(rustc_attrs)]

#[rustc_scalable_vector(2)]
struct ValidI64(i64);

#[rustc_scalable_vector(4)]
struct ValidI32(i32);

#[rustc_scalable_vector]
struct ValidTuple(ValidI32, ValidI32, ValidI32);

#[rustc_scalable_vector]
struct Struct { x: ValidI64, y: ValidI64 }
//~^ ERROR: scalable vectors must be tuple structs

#[rustc_scalable_vector]
struct DifferentVectorTypes(ValidI64, ValidI32);
//~^ ERROR: all fields in a scalable vector struct must be the same type

#[rustc_scalable_vector]
struct NonVectorTypes(u32, u64);
//~^ ERROR: scalable vector structs can only have scalable vector fields

#[rustc_scalable_vector]
struct DifferentNonVectorTypes(u32, u64);
//~^ ERROR: scalable vector structs can only have scalable vector fields

#[rustc_scalable_vector]
struct SomeVectorTypes(ValidI64, u64);
//~^ ERROR: scalable vector structs can only have scalable vector fields

#[rustc_scalable_vector]
struct NestedTuple(ValidTuple, ValidTuple);
//~^ ERROR: scalable vector structs cannot contain other scalable vector structs
