//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::field_of;

pub struct Struct {
    field: i32,
}

pub union Union {
    field: i32,
}

pub enum Enum {
    Variant { field: i32 },
}

pub trait Trait {}

fn main() {
    let _: field_of!(Struct, other); //~ ERROR: no field `other` on struct `Struct` [E0609]
    let _: field_of!(Struct, 0); //~ ERROR: no field `0` on struct `Struct` [E0609]
    let _: field_of!(Union, other); //~ ERROR: no field `other` on union `Union` [E0609]
    let _: field_of!(Union, 0); //~ ERROR: no field `0` on union `Union` [E0609]
    // FIXME(FRTs): make the error mention the variant too.
    let _: field_of!(Enum, Variant.other); //~ ERROR: no field `other` on enum `Enum` [E0609]
    let _: field_of!(Enum, Variant.0); //~ ERROR: no field `0` on enum `Enum` [E0609]
    // FIXME(FRTs): select correct error code
    let _: field_of!(Enum, OtherVariant.field); //~ ERROR: no variant `OtherVariant` on enum `Enum`
    let _: field_of!(Enum, OtherVariant.0); //~ ERROR: no variant `OtherVariant` on enum `Enum`

    let _: field_of!(i32, field); //~ ERROR: type `i32` doesn't have fields
    let _: field_of!([Struct], field); //~ ERROR: type `[Struct]` doesn't have fields
    let _: field_of!([Struct; 42], field); //~ ERROR: type `[Struct; 42]` is not yet supported in `field_of!`
    let _: field_of!(&'static Struct, field); //~ ERROR: type `&'static Struct` doesn't have fields
    let _: field_of!(*const Struct, field); //~ ERROR: type `*const Struct` doesn't have fields
    let _: field_of!(fn() -> Struct, field); //~ ERROR: type `fn() -> Struct` doesn't have fields
    let _: field_of!(dyn Trait, field); //~ ERROR: type `dyn Trait` doesn't have fields
    let _: field_of!(main, field); //~ ERROR: expected type, found function `main`
    let _: field_of!(_, field); //~ ERROR: cannot use `_` in this position
}

fn generic<T>() {
    let _: field_of!(T, field); //~ ERROR: type `T` doesn't have fields
}
