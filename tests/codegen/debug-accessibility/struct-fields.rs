//@ compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for struct fields.

mod module {
    use std::hint::black_box;

    struct StructFields {
        a: u32,
        pub(crate) b: u32,
        pub(super) c: u32,
        pub d: u32,
    }

    // CHECK: [[StructFields:!.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "StructFields"{{.*}}flags: DIFlagPrivate{{.*}})
    // CHECK: {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: [[StructFields]]{{.*}}flags: DIFlagPrivate{{.*}})
    // CHECK: {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: [[StructFields]]{{.*}}flags: DIFlagProtected{{.*}})
    // CHECK: {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: [[StructFields]]{{.*}}flags: DIFlagProtected{{.*}})
    // CHECK: {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: [[StructFields]]{{.*}}flags: DIFlagPublic{{.*}})

    pub fn use_everything() {
        black_box(StructFields { a: 1, b: 2, c: 3, d: 4 });
    }
}

fn main() {
    module::use_everything();
}
