// ignore-windows
//min-system-llvm-version 8.0

//compile-flags: -g -C no-prepopulate-passes

// === GDB TESTS ===================================================================================

// gdb-command: run


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
