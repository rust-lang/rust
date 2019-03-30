// compile-flags: --emit=llvm-ir -C debuginfo=2
#![feature(repr128)]

#[repr(u128)]
pub enum Foo {
    Lo,
    Hi = 1 << 64,
}

pub fn foo() -> Option<Foo> {
    None
}

// CHECK: declare void @llvm.dbg.value
fn main() {
    let vals = (Some(Foo::Lo), None::<Foo>);
}
