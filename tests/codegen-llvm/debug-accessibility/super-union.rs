//@ compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for super-visibility unions.

mod module {
    use std::hint::black_box;

    pub(super) union SuperFooUnion {
        x: u32,
    }

    // CHECK: {{!.*}} = !DICompositeType(tag: DW_TAG_union_type, name: "SuperFooUnion"{{.*}}flags: DIFlagProtected{{.*}})

    pub fn use_everything() {
        black_box(SuperFooUnion { x: 3 });
    }
}

fn main() {
    module::use_everything();
}
