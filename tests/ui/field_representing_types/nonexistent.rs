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
}
