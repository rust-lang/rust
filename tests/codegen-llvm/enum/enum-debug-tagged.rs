// This tests that debug info for tagged (ordinary) enums is properly emitted.
// This is ignored for the fallback mode on MSVC due to problems with PDB.

//@ ignore-msvc
//@ ignore-wasi wasi codegens the main symbol differently

//@ compile-flags: -g -C no-prepopulate-passes

// CHECK-LABEL: @main
// CHECK-DAG: {{.*}}DICompositeType{{.*}}tag: DW_TAG_structure_type,{{.*}}name: "E",{{.*}}
// CHECK-DAG: {{.*}}DICompositeType{{.*}}tag: DW_TAG_variant_part,{{.*}}discriminator:{{.*}}
// CHECK-DAG: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "A",{{.*}}extraData:{{.*}}
// CHECK-DAG: {{.*}}DICompositeType{{.*}}tag: DW_TAG_structure_type,{{.*}}name: "A",{{.*}}
// CHECK-DAG: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "__0",{{.*}}
// CHECK-DAG: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "B",{{.*}}extraData:{{.*}}
// CHECK-DAG: {{.*}}DICompositeType{{.*}}tag: DW_TAG_structure_type,{{.*}}name: "B",{{.*}}
// CHECK-DAG: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "__0",{{.*}}
// CHECK-DAG: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}flags: DIFlagArtificial{{.*}}

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

enum E {
    A(u32),
    B(u32),
}

pub fn main() {
    let e = E::A(23);
}
