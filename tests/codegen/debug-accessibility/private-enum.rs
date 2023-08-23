// compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for private enums.

use std::hint::black_box;

enum PrivateFooEnum {
    A,
    B(u32),
    C { x: u32 },
}

// CHECK: {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "PrivateFooEnum"{{.*}}flags: DIFlagPrivate{{.*}})

fn main() {
    black_box(PrivateFooEnum::A);
}
