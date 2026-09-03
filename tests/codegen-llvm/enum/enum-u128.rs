// This tests that debug info for "c-like" 128bit enums is properly emitted.
// This is ignored for the fallback mode on MSVC due to problems with PDB.

//
//@ ignore-msvc
//@ ignore-wasi wasi codegens the main symbol differently

//@ compile-flags: -g -C no-prepopulate-passes

// CHECK-LABEL: @main
// CHECK-DAG: {{.*}}DICompositeType{{.*}}tag: DW_TAG_enumeration_type,{{.*}}name: "Foo",{{.*}}flags: DIFlagEnumClass,{{.*}}
// CHECK-DAG: {{.*}}DIEnumerator{{.*}}name: "Lo",{{.*}}value: 0,{{.*}}
// CHECK-DAG: {{.*}}DIEnumerator{{.*}}name: "Hi",{{.*}}value: 18446744073709551616,{{.*}}
// CHECK-DAG: {{.*}}DIEnumerator{{.*}}name: "Bar",{{.*}}value: 18446745000000000123,{{.*}}

#[repr(u128)]
pub enum Foo {
    Lo,
    Hi = 1 << 64,
    Bar = 18_446_745_000_000_000_123,
}

pub fn main() {
    let foo = Foo::Bar;
}
