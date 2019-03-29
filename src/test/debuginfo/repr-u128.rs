// ignore-windows
// ignore-tidy-linelength
// min-system-llvm-version 8.0

// compile-flags: -g -C no-prepopulate-passes

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command:print vals
// gdb-check:$1 = (core::option::Option<repr_u128::Foo>, core::option::Option<repr_u128::Foo>)

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print vals
// lldbg-check:[...]$0 = (Option<repr_u128::Foo> { }, Option<repr_u128::Foo> { })
// lldbr-check:((core::option::Option<repr_u128::Foo>, core::option::Option<repr_u128::Foo>)) $0 = (Option<repr_u128::Foo> { }, Option<repr_u128::Foo> { })

#![feature(repr128)]

#[repr(u128)]
pub enum Foo {
    Lo,
    Hi = 1 << 64,
    Bar = 18_446_745_000_000_000_123,
}

fn main() {
    let vals = (Some(Foo::Lo), None::<Foo>);

    zzz(); // #break
}

fn zzz() {()}
