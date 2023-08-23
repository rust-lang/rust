// compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for super-visibility enums.

mod module {
    use std::hint::black_box;

    pub(super) enum SuperFooEnum {
        A,
        B(u32),
        C { x: u32 },
    }

    // CHECK: {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "SuperFooEnum"{{.*}}flags: DIFlagProtected{{.*}})

    pub fn use_everything() {
        black_box(SuperFooEnum::A);
    }
}

fn main() {
    module::use_everything();
}
