// compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for types and their fields.

use std::hint::black_box;

pub enum PublicFooEnum {
    A,
    B(u32),
    C { x: u32 },
}

// CHECK: {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "PublicFooEnum"{{.*}}flags: DIFlagPublic{{.*}})

fn main() {
    black_box(PublicFooEnum::A);
}
