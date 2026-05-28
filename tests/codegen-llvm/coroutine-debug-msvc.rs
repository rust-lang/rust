// Verify debuginfo for coroutines:
//  - Each variant points to the file and line of its yield point
//  - The discriminants are marked artificial
//  - Other fields are not marked artificial
//
//
//@ compile-flags: -C debuginfo=2
//@ only-msvc

#![feature(coroutines, coroutine_trait)]
use std::ops::Coroutine;

fn coroutine_test() -> impl Coroutine<Yield = i32, Return = ()> {
    #[coroutine]
    || {
        yield 0;
        let s = String::from("foo");
        yield 1;
    }
}

// FIXME: No way to reliably check the filename.

// CHECK-DAG:  [[GEN:!.*]] = !DICompositeType(tag: DW_TAG_union_type, name: "enum2$<coroutine_debug_msvc::coroutine_test::coroutine_env$0>"
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "variant0", scope: [[GEN]],
// For brevity, we only check the struct name and members of the last variant.
// CHECK-SAME: file: [[FILE:![0-9]*]], line: 15,
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "variant1", scope: [[GEN]],
// CHECK-SAME: file: [[FILE]], line: 19,
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "variant2", scope: [[GEN]],
// CHECK-SAME: file: [[FILE]], line: 19,
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "variant3", scope: [[GEN]],
// CHECK-SAME: file: [[FILE]], line: 16,
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "variant4", scope: [[GEN]],
// CHECK-SAME: file: [[FILE]], line: 18,
// CHECK-SAME: baseType: [[VARIANT_WRAPPER:![0-9]*]]
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      [[VARIANT_WRAPPER]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Variant4", scope: [[GEN]],
// CHECK:      !DIDerivedType(tag: DW_TAG_member, name: "value", scope: [[VARIANT_WRAPPER]], {{.*}}, baseType: [[VARIANT:![0-9]*]],
// CHECK:      [[VARIANT]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Suspend1", scope: [[GEN]],
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "s", scope: [[VARIANT]]
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "tag", scope: [[GEN]],
// CHECK-NOT: flags: DIFlagArtificial

fn main() {
    let _dummy = coroutine_test();
}
