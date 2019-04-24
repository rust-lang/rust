// ignore-windows
//min-system-llvm-version 8.0

//compile-flags: -g -C no-prepopulate-passes
#![feature(repr128)]

#[repr(u128)]
pub enum Foo {
    Lo,
    Hi = 1 << 64,
}

fn main() {
    let vals = (Some(Foo::Lo), None::<Foo>);
}
