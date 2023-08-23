// compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for crate-visibility enums.

mod module {
    use std::hint::black_box;

    pub(crate) enum CrateFooEnum {
        A,
        B(u32),
        C { x: u32 },
    }

    // CHECK: {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "CrateFooEnum"{{.*}}flags: DIFlagProtected{{.*}})
    pub fn use_everything() {
        black_box(CrateFooEnum::A);
    }
}

fn main() {
    module::use_everything();
}
