// ignore-windows
//min-system-llvm-version 8.0

//compile-flags: -g -C no-prepopulate-passes
#![feature(repr128)]

#[repr(u128)]
pub enum Foo {
    Lo,
    Hi = 1 << 64,
    Bar = 18_446_745_000_000_000_123,
}

// CHECK: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "None",{{.*}}extraData:{{.*}}
pub fn foo() -> Option<Foo> {
    None
}

// CHECK: declare void @llvm.dbg.value
fn main() {}
