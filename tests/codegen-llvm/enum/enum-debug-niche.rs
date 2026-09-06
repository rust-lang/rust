// This tests that optimized enum debug info accurately reflects the enum layout.
// This is ignored for the fallback mode on MSVC due to problems with PDB.

//@ ignore-msvc
//@ ignore-wasi wasi codegens the main symbol differently

//@ compile-flags: -g -C no-prepopulate-passes

// CHECK-LABEL: @main
// CHECK-DAG: {{.*}}DICompositeType{{.*}}tag: DW_TAG_variant_part,{{.*}}discriminator:{{.*}}
// CHECK-DAG: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "A",{{.*}}extraData:{{.*}}
// CHECK-DAG: {{.*}}DICompositeType{{.*}}tag: DW_TAG_structure_type,{{.*}}name: "A",{{.*}}
// CHECK-DAG: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "B",{{.*}}extraData:{{.*}}
// CHECK-DAG: {{.*}}DICompositeType{{.*}}tag: DW_TAG_structure_type,{{.*}}name: "B",{{.*}}
// CHECK-DAG: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "C",{{.*}}extraData:{{.*}}
// CHECK-DAG: {{.*}}DICompositeType{{.*}}tag: DW_TAG_structure_type,{{.*}}name: "C",{{.*}}
// CHECK-DAG: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "D",{{.*}}
// CHECK-DAG: {{.*}}DICompositeType{{.*}}tag: DW_TAG_structure_type,{{.*}}name: "D",{{.*}}
// CHECK-DAG: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}flags: DIFlagArtificial{{.*}}
// CHECK-NOT: {{.*}}DIDerivedType{{.*}}name: "D",{{.*}}extraData:{{.*}}

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

enum E {
    A,
    B,
    C,
    D(bool),
}

pub fn main() {
    let e = E::D(true);
}
