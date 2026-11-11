// This tests that debug info for "c-like" enums is properly emitted.
// This is ignored for the fallback mode on MSVC due to problems with PDB.

//
//@ ignore-msvc
//@ ignore-wasi wasi codegens the main symbol differently

//@ compile-flags: -g -C no-prepopulate-passes

// CHECK-LABEL: @main
// CHECK: {{.*}}DICompositeType{{.*}}tag: DW_TAG_enumeration_type,{{.*}}name: "E",{{.*}}flags: DIFlagEnumClass,{{.*}}
// CHECK: {{.*}}DIEnumerator{{.*}}name: "A",{{.*}}value: {{[0-9].*}}
// CHECK: {{.*}}DIEnumerator{{.*}}name: "B",{{.*}}value: {{[0-9].*}}
// CHECK: {{.*}}DIEnumerator{{.*}}name: "C",{{.*}}value: {{[0-9].*}}

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

enum E {
    A,
    B,
    C,
}

pub fn main() {
    let e = E::C;
}
