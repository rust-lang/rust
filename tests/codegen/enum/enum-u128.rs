// This tests that debug info for "c-like" 128bit enums is properly emitted.
// This is ignored for the fallback mode on MSVC due to problems with PDB.

//
// ignore-msvc

// compile-flags: -g -C no-prepopulate-passes

// CHECK-LABEL: @main
// CHECK: {{.*}}DICompositeType{{.*}}tag: DW_TAG_enumeration_type,{{.*}}name: "Foo",{{.*}}flags: DIFlagEnumClass,{{.*}}
// CHECK: {{.*}}DIEnumerator{{.*}}name: "Lo",{{.*}}value: 0,{{.*}}
// CHECK: {{.*}}DIEnumerator{{.*}}name: "Hi",{{.*}}value: 18446744073709551616,{{.*}}
// CHECK: {{.*}}DIEnumerator{{.*}}name: "Bar",{{.*}}value: 18446745000000000123,{{.*}}

#![allow(incomplete_features)]
#![feature(repr128)]

#[repr(u128)]
pub enum Foo {
    Lo,
    Hi = 1 << 64,
    Bar = 18_446_745_000_000_000_123,
}

pub fn main() {
    let foo = Foo::Bar;
}
