// ignore-tidy-linelength
//! Checks that visibility information is present in the debuginfo for super-visibility enums.

//@ revisions: MSVC NONMSVC
//@[MSVC] only-msvc
//@[NONMSVC] ignore-msvc
//@ compile-flags: -C debuginfo=2

mod module {
    use std::hint::black_box;

    pub(super) enum SuperFooEnum {
        A,
        B(u32),
        C { x: u32 },
    }

    // NONMSVC: {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "SuperFooEnum"{{.*}}flags: DIFlagProtected{{.*}})
    // MSVC: {{!.*}} = !DICompositeType(tag: DW_TAG_union_type, name: "enum2$<super_enum::module::SuperFooEnum>"{{.*}}flags: DIFlagProtected{{.*}})

    pub fn use_everything() {
        black_box(SuperFooEnum::A);
    }
}

fn main() {
    module::use_everything();
}
