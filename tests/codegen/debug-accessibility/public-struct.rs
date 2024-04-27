//@ compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for public structs.

use std::hint::black_box;

pub struct PublicFooStruct {
    x: u32,
}

// CHECK: {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "PublicFooStruct"{{.*}}flags: DIFlagPublic{{.*}})

fn main() {
    black_box(PublicFooStruct { x: 4 });
}
