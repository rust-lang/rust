//@ compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for private structs.

use std::hint::black_box;

struct PrivateFooStruct {
    x: u32,
}

// CHECK: {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "PrivateFooStruct"{{.*}}flags: DIFlagPrivate{{.*}})

fn main() {
    black_box(PrivateFooStruct { x: 1 });
}
