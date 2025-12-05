//@ run-pass

#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

pub trait Foo {
    fn foo();
}

impl Foo for i32 {}
impl Foo for i64 {}
impl<T> Foo for T {
    fn foo() {}
}

fn main() {
    i32::foo();
    i64::foo();
    u8::foo();
}
