// ignore-windows
// ignore-tidy-linelength
//min-system-llvm-version 8.0

//compile-flags: -g -C no-prepopulate-passes
#![feature(repr128)]

#[repr(u128)]
pub enum Foo {
    Lo,
    Hi = 1 << 64,
    Bar = 18_446_745_000_000_000_123,
}

// CHECK: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "None",{{.*}}extraData:18446745000000000124
pub fn foo() -> Option<Foo> {
    None
}

fn main() {}
