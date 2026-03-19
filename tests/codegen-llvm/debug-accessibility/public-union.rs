//@ compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for public unions.

use std::hint::black_box;

pub union PublicFooUnion {
    x: u32,
}

// CHECK: {{!.*}} = !DICompositeType(tag: DW_TAG_union_type, name: "PublicFooUnion"{{.*}}flags: DIFlagPublic{{.*}})

fn main() {
    black_box(PublicFooUnion { x: 4 });
}
