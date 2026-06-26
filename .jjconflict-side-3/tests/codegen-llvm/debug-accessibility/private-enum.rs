// ignore-tidy-linelength
//! Checks that visibility information is present in the debuginfo for private enums.

//@ revisions: MSVC NONMSVC
//@[MSVC] only-msvc
//@[NONMSVC] ignore-msvc
//@ compile-flags: -C debuginfo=2

use std::hint::black_box;

enum PrivateFooEnum {
    A,
    B(u32),
    C { x: u32 },
}

// NONMSVC: {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "PrivateFooEnum"{{.*}}flags: DIFlagPrivate{{.*}})
// MSVC: {{!.*}} = !DICompositeType(tag: DW_TAG_union_type, name: "enum2$<private_enum::PrivateFooEnum>"{{.*}}flags: DIFlagPrivate{{.*}})

fn main() {
    black_box(PrivateFooEnum::A);
}
