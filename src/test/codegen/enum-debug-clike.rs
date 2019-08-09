// This test depends on a patch that was committed to upstream LLVM
// before 7.0, then backported to the Rust LLVM fork.  It tests that
// debug info for "c-like" enums is properly emitted.

// ignore-tidy-linelength
// ignore-windows
// min-system-llvm-version 8.0

// compile-flags: -g -C no-prepopulate-passes

// DIFlagFixedEnum was deprecated in 8.0, renamed to DIFlagEnumClass.
// We match either for compatibility.

// CHECK-LABEL: @main
// CHECK: {{.*}}DICompositeType{{.*}}tag: DW_TAG_enumeration_type,{{.*}}name: "E",{{.*}}flags: {{(DIFlagEnumClass|DIFlagFixedEnum)}},{{.*}}
// CHECK: {{.*}}DIEnumerator{{.*}}name: "A",{{.*}}value: {{[0-9].*}}
// CHECK: {{.*}}DIEnumerator{{.*}}name: "B",{{.*}}value: {{[0-9].*}}
// CHECK: {{.*}}DIEnumerator{{.*}}name: "C",{{.*}}value: {{[0-9].*}}

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

enum E { A, B, C }

pub fn main() {
    let e = E::C;
}
