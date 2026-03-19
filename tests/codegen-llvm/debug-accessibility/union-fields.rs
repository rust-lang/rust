//@ compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for union fields.

mod module {
    use std::hint::black_box;

    union UnionFields {
        a: u32,
        pub(crate) b: u32,
        pub(super) c: u32,
        pub d: u32,
    }

    // CHECK: [[UnionFields:!.*]] = !DICompositeType(tag: DW_TAG_union_type, name: "UnionFields"{{.*}}flags: DIFlagPrivate{{.*}})
    // CHECK: {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: [[UnionFields]]{{.*}}flags: DIFlagPrivate{{.*}})
    // CHECK: {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: [[UnionFields]]{{.*}}flags: DIFlagProtected{{.*}})
    // CHECK: {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: [[UnionFields]]{{.*}}flags: DIFlagProtected{{.*}})
    // CHECK: {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: [[UnionFields]]{{.*}}flags: DIFlagPublic{{.*}})

    pub fn use_everything() {
        black_box(UnionFields { a: 1 });
    }
}

fn main() {
    module::use_everything();
}
