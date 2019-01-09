// compile-fail

#![feature(specialization)]

pub trait Foo {
    fn foo();
}

impl Foo for i32 {}
impl Foo for i64 {
    fn foo() {}
    //~^ERROR `foo` specializes an item from a parent `impl`
}
impl<T> Foo for T {
    fn foo() {}
}

fn main() {
    i32::foo();
    i64::foo();
    u8::foo();
}
