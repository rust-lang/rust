//@ compile-flags: -C debuginfo=2
// ignore-tidy-linelength

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for types and their fields.

use std::hint::black_box;

pub enum PublicFooEnum {
    A,
    B(u32),
    C { x: u32 },
}

// NONMSVC: {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "PublicFooEnum"{{.*}}flags: DIFlagPublic{{.*}})
// MSVC: {{!.*}} = !DICompositeType(tag: DW_TAG_union_type, name: "enum2$<public_enum::PublicFooEnum>"{{.*}}flags: DIFlagPublic{{.*}})

fn main() {
    black_box(PublicFooEnum::A);
}
