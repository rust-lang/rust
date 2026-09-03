// Verify debuginfo for coroutines:
//  - Each variant points to the file and line of its yield point
//  - The discriminants are marked artificial
//  - Other fields are not marked artificial
//
//
//@ compile-flags: -C debuginfo=2
//@ edition: 2018
//@ ignore-msvc

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

// CHECK-DAG:  [[GEN_FN:!.*]] = !DINamespace(name: "coroutine_test"
// CHECK-DAG:  [[GEN:!.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "{coroutine_env#0}", scope: [[GEN_FN]]
// CHECK-DAG:  [[VARIANT:!.*]] = !DICompositeType(tag: DW_TAG_variant_part, scope: [[GEN]], {{.*}}discriminator: [[DISC:![0-9]*]]
// CHECK-DAG:  {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "0", scope: [[VARIANT]], {{.*}}file: [[FILE:![0-9]*]], line: 16,
// CHECK-DAG:  {{!.*}} = !DICompositeType(tag: DW_TAG_structure_type, name: "Unresumed", scope: [[GEN]],
// CHECK-DAG:  {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "1", scope: [[VARIANT]], {{.*}}file: [[FILE]], line: 20,
// CHECK-DAG:  {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "2", scope: [[VARIANT]], {{.*}}file: [[FILE]], line: 20,
// CHECK-DAG:  {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "3", scope: [[VARIANT]], {{.*}}file: [[FILE]], line: 17,
// CHECK-DAG:  {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "4", scope: [[VARIANT]], {{.*}}file: [[FILE]], line: 19,
// CHECK-DAG:  [[S1:!.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Suspend1", scope: [[GEN]],
// CHECK-DAG:  {{!.*}} = !DIDerivedType(tag: DW_TAG_member, name: "s", scope: [[S1]]
// CHECK-DAG:  [[DISC]] = !DIDerivedType(tag: DW_TAG_member, name: "__state", scope: [[GEN]],

fn main() {
    let _dummy = coroutine_test();
}
