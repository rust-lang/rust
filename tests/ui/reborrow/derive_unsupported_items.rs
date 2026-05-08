#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

#[derive(Reborrow)] //~ ERROR `derive(Reborrow)` is only supported for structs, not enums
enum ReborrowEnum<'a> {
    Variant(&'a mut ()),
}

#[derive(CoerceShared)] //~ ERROR `derive(CoerceShared)` is only supported for structs, not enums
#[coerce_shared(CoerceSharedEnumRef<'a>)]
enum CoerceSharedEnum<'a> {
    Variant(&'a mut ()),
}

#[derive(Clone, Copy)]
struct CoerceSharedEnumRef<'a>(&'a ());

#[derive(Reborrow)] //~ ERROR `derive(Reborrow)` is only supported for structs, not unions
union ReborrowUnion<'a> {
    field: &'a mut (),
}

#[derive(CoerceShared)] //~ ERROR `derive(CoerceShared)` is only supported for structs, not unions
#[coerce_shared(CoerceSharedUnionRef<'a>)]
union CoerceSharedUnion<'a> {
    field: &'a mut (),
}

#[derive(Clone, Copy)]
struct CoerceSharedUnionRef<'a>(&'a ());

fn main() {}
