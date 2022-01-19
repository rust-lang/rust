// Verify debuginfo for generators:
//  - Each variant points to the file and line of its yield point
//  - The discriminants are marked artificial
//  - Other fields are not marked artificial
//
//
// compile-flags: -C debuginfo=2
// only-msvc

#![feature(generators, generator_trait)]
use std::ops::Generator;

fn generator_test() -> impl Generator<Yield = i32, Return = ()> {
    || {
        yield 0;
        let s = String::from("foo");
        yield 1;
    }
}

// FIXME: No way to reliably check the filename.

// CHECK-DAG:  [[GEN_FN:!.*]] = !DINamespace(name: "generator_test"
// CHECK-DAG:  [[GEN:!.*]] = !DICompositeType(tag: DW_TAG_union_type, name: "generator_env$0"
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "variant0", scope: [[GEN]],
// For brevity, we only check the struct name and members of the last variant.
// CHECK-SAME: file: [[FILE:![0-9]*]], line: 14,
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "variant1", scope: [[GEN]],
// CHECK-SAME: file: [[FILE]], line: 18,
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "variant2", scope: [[GEN]],
// CHECK-SAME: file: [[FILE]], line: 18,
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "variant3", scope: [[GEN]],
// CHECK-SAME: file: [[FILE]], line: 15,
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "variant4", scope: [[GEN]],
// CHECK-SAME: file: [[FILE]], line: 17,
// CHECK-SAME: baseType: [[VARIANT:![0-9]*]]
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      [[S1:!.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Suspend1", scope: [[GEN]],
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "s", scope: [[S1]]
// CHECK-NOT:  flags: DIFlagArtificial
// CHECK-SAME: )
// CHECK:      {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "discriminant", scope: [[GEN]],
// CHECK-NOT: flags: DIFlagArtificial

fn main() {
    let _dummy = generator_test();
}
