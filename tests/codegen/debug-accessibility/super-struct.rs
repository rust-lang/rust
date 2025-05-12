//@ compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for super-visibility structs.

mod module {
    use std::hint::black_box;

    pub(super) struct SuperFooStruct {
        x: u32,
    }

    // CHECK: {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "SuperFooStruct"{{.*}}flags: DIFlagProtected{{.*}})

    pub fn use_everything() {
        black_box(SuperFooStruct { x: 3 });
    }
}

fn main() {
    module::use_everything();
}
