//@ compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for crate-visibility structs.

mod module {
    use std::hint::black_box;

    pub(crate) struct CrateFooStruct {
        x: u32,
    }

    // CHECK: {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "CrateFooStruct"{{.*}}flags: DIFlagProtected{{.*}})

    pub fn use_everything() {
        black_box(CrateFooStruct { x: 2 });
    }
}

fn main() {
    module::use_everything();
}
