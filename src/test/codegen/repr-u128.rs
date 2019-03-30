// compile-flags: --emit=llvm-ir -C debuginfo=2
#![feature(repr128)]

#[repr(u128)]
pub enum Foo {
    Lo,
    Hi = 1 << 64,
    Bar = 18_446_745_000_000_000_123,
}

pub fn foo() -> Option<Foo> {
    None
}

// CHECK: declare void @llvm.dbg.value
fn main() {}
