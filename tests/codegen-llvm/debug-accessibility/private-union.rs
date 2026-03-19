//@ compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for private unions.

use std::hint::black_box;

union PrivateFooUnion {
    x: u32,
}

// CHECK: {{!.*}} = !DICompositeType(tag: DW_TAG_union_type, name: "PrivateFooUnion"{{.*}}flags: DIFlagPrivate{{.*}})

fn main() {
    black_box(PrivateFooUnion { x: 1 });
}
