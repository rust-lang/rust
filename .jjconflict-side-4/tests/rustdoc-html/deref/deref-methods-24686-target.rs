#![crate_name = "foo"]

// test for https://github.com/rust-lang/rust/issues/24686
use std::ops::Deref;

pub struct Foo<T>(T);
impl Foo<i32> {
    pub fn get_i32(&self) -> i32 { self.0 }
}
impl Foo<u32> {
    pub fn get_u32(&self) -> u32 { self.0 }
}

// Note that the same href is used both on the method itself,
// and on the sidebar items.
//@ has foo/struct.Bar.html
//@ has - '//a[@href="#method.get_i32"]' 'get_i32'
//@ !has - '//a[@href="#method.get_u32"]' 'get_u32'
//@ count - '//ul[@class="block deref-methods"]//a' 1
//@ count - '//a[@href="#method.get_i32"]' 2
pub struct Bar(Foo<i32>);
impl Deref for Bar {
    type Target = Foo<i32>;
    fn deref(&self) -> &Foo<i32> {
        &self.0
    }
}
