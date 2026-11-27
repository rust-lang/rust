//! Test that deriving Reborrow and CoerceShared errors on enums and unions.

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

#[derive(Reborrow)]
enum ReborrowEnum<'a> {
    Variant(&'a mut ()),
}

#[derive(CoerceShared)] //~ ERROR the trait `CoerceShared` may only be implemented for a coercion between structures
#[coerce_shared(CoerceSharedEnumRef<'a>)]
enum CoerceSharedEnum<'a> {
    Variant(&'a mut ()),
}

#[derive(Clone, Copy)]
struct CoerceSharedEnumRef<'a>(&'a ());

#[derive(Reborrow)] //~ ERROR `derive(Reborrow)` is only supported for structs or enums, not unions
union ReborrowUnion<'a> {
    field: &'a mut (),
}

#[derive(CoerceShared)] //~ ERROR `derive(CoerceShared)` is only supported for structs or enums, not unions
#[coerce_shared(CoerceSharedUnionRef<'a>)]
union CoerceSharedUnion<'a> {
    field: &'a mut (),
}

#[derive(Clone, Copy)]
struct CoerceSharedUnionRef<'a>(&'a ());

fn main() {}
