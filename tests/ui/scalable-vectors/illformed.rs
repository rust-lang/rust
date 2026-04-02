//@ compile-flags: --crate-type=lib
//@ only-aarch64
#![allow(internal_features)]
#![feature(rustc_attrs)]

#[rustc_scalable_vector(4)]
struct Valid(f32);

#[rustc_scalable_vector(4)]
struct NoFieldsStructWithElementCount {}
//~^ ERROR: scalable vectors must have a single field
//~^^ ERROR: scalable vectors must be tuple structs

#[rustc_scalable_vector(4)]
struct NoFieldsTupleWithElementCount();
//~^ ERROR: scalable vectors must have a single field

#[rustc_scalable_vector(4)]
struct NoFieldsUnitWithElementCount;
//~^ ERROR: scalable vectors must have a single field
//~^^ ERROR: scalable vectors must be tuple structs

#[rustc_scalable_vector]
struct NoFieldsStructWithoutElementCount {}
//~^ ERROR: scalable vector tuples must have at least one field
//~^^ ERROR: scalable vectors must be tuple structs

#[rustc_scalable_vector]
struct NoFieldsTupleWithoutElementCount();
//~^ ERROR: scalable vector tuples must have at least one field

#[rustc_scalable_vector]
struct NoFieldsUnitWithoutElementCount;
//~^ ERROR: scalable vector tuples must have at least one field
//~^^ ERROR: scalable vectors must be tuple structs

#[rustc_scalable_vector(4)]
struct MultipleFieldsStructWithElementCount {
//~^ ERROR: scalable vectors cannot have multiple fields
//~^^ ERROR: scalable vectors must be tuple structs
    _ty: f32,
    other: u32,
}

#[rustc_scalable_vector(4)]
struct MultipleFieldsTupleWithElementCount(f32, u32);
//~^ ERROR: scalable vectors cannot have multiple fields

#[rustc_scalable_vector]
struct MultipleFieldsStructWithoutElementCount {
//~^ ERROR: scalable vectors must be tuple structs
    _ty: f32,
//~^ ERROR: scalable vector structs can only have scalable vector fields
    other: u32,
}

#[rustc_scalable_vector]
struct MultipleFieldsTupleWithoutElementCount(f32, u32);
//~^ ERROR: scalable vector structs can only have scalable vector fields

#[rustc_scalable_vector(2)]
struct SingleFieldStruct { _ty: f64 }
//~^ ERROR: scalable vectors must be tuple structs

#[rustc_scalable_vector]
struct TooManyFieldsWithoutElementCount(
    Valid, Valid, Valid, Valid, Valid, Valid, Valid, Valid, Valid);
//~^^ ERROR: scalable vector tuples can have at most eight fields
