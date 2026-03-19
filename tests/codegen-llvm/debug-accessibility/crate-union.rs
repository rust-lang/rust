//@ compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for crate-visibility unions.

mod module {
    use std::hint::black_box;

    pub(crate) union CrateFooUnion {
        x: u32,
    }

    // CHECK: {{!.*}} = !DICompositeType(tag: DW_TAG_union_type, name: "CrateFooUnion"{{.*}}flags: DIFlagProtected{{.*}})

    pub fn use_everything() {
        black_box(CrateFooUnion { x: 2 });
    }
}

fn main() {
    module::use_everything();
}
