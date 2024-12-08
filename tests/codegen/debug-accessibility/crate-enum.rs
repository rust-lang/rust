//@ compile-flags: -C debuginfo=2
// ignore-tidy-linelength

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for crate-visibility enums.

mod module {
    use std::hint::black_box;

    pub(crate) enum CrateFooEnum {
        A,
        B(u32),
        C { x: u32 },
    }

    // NONMSVC: {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "CrateFooEnum"{{.*}}flags: DIFlagProtected{{.*}})
    // MSVC: {{!.*}} = !DICompositeType(tag: DW_TAG_union_type, name: "enum2$<crate_enum::module::CrateFooEnum>"{{.*}}flags: DIFlagProtected{{.*}})
    pub fn use_everything() {
        black_box(CrateFooEnum::A);
    }
}

fn main() {
    module::use_everything();
}
