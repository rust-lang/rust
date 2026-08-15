//@ compile-flags: -C debuginfo=2

#![allow(dead_code)]

// Checks that visibility information is present in the debuginfo for tuple struct fields.

mod module {
    use std::hint::black_box;

    struct TupleFields(u32, pub(crate) u32, pub(super) u32, pub u32);

    // CHECK: [[TupleFields:!.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "TupleFields"{{.*}}flags: DIFlagPrivate{{.*}})
    // CHECK: {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: [[TupleFields]]{{.*}}flags: DIFlagPrivate{{.*}})
    // CHECK: {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "__1", scope: [[TupleFields]]{{.*}}flags: DIFlagProtected{{.*}})
    // CHECK: {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "__2", scope: [[TupleFields]]{{.*}}flags: DIFlagProtected{{.*}})
    // CHECK: {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "__3", scope: [[TupleFields]]{{.*}}flags: DIFlagPublic{{.*}})
    pub fn use_everything() {
        black_box(TupleFields(1, 2, 3, 4));
    }
}

fn main() {
    module::use_everything();
}
