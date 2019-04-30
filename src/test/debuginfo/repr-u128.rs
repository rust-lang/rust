// ignore-windows
// ignore-tidy-linelength
//min-system-llvm-version 8.0

//compile-flags: -g -C no-prepopulate-passes

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command:print vals
// gdbg-check:$1 = (Some(Foo::Lo), None::<Foo>)
// gdbr-check:$1 = repr_u128::Foo::(std::option::Option<Foo>, std::option::Option<Foo>)

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print vals
// lldbg-check:[...]$0 = (Some(Foo::Lo), None::<Foo>)
// lldbr-check:(repr_u128::Foo) vals = repr_u128::Foo::(std::option::Option<Foo>, std::option::Option<Foo>)

#![feature(repr128)]

#[repr(u128)]
pub enum Foo {
    Lo,
    Hi = 1 << 64,
    Bar = 18_446_745_000_000_000_123,
}

fn main() {
    let vals = (Some(Foo::Lo), None::<Foo>);
}
