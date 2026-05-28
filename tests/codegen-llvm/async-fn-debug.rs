// Verify debuginfo for async fn:
//  - Each variant points to the file and line of its yield point
//  - The discriminants are marked artificial
//  - Other fields are not marked artificial
//
//
//@ compile-flags: -C debuginfo=2
//@ edition: 2018
//@ ignore-msvc

async fn foo() {}
async fn async_fn_test() {
    foo().await;
    let s = String::from("foo");
    foo().await;
}

// FIXME: No way to reliably check the filename.

// CHECK-DAG:  [[ASYNC_FN:!.*]] = !DINamespace(name: "async_fn_test"
// CHECK-DAG:  [[GEN:!.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "{async_fn_env#0}", scope: [[ASYNC_FN]]
// CHECK:      [[VARIANT:!.*]] = !DICompositeType(tag: DW_TAG_variant_part, scope: [[GEN]],
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: discriminator: [[DISC:![0-9]*]]
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "0", scope: [[VARIANT]],
// CHECK-SAME: file: [[FILE:![0-9]*]], line: 12,
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "Unresumed", scope: [[GEN]],
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "1", scope: [[VARIANT]],
// CHECK-SAME: file: [[FILE]], line: 16,
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "2", scope: [[VARIANT]],
// CHECK-SAME: file: [[FILE]], line: 16,
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "3", scope: [[VARIANT]],
// CHECK-SAME: file: [[FILE]], line: 13,
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "4", scope: [[VARIANT]],
// CHECK-SAME: file: [[FILE]], line: 15,
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      [[S1:!.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Suspend1", scope: [[GEN]],
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "s", scope: [[S1]]
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      [[DISC]] = !DIDerivedType(tag: DW_TAG_member, name: "__state", scope: [[GEN]],
// CHECK-SAME: flags: DIFlagArtificial

fn main() {
    let _dummy = async_fn_test();
}
