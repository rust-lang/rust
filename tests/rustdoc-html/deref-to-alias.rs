// This test ensures that dereferencing to a type alias will still list
// matching methods of the type alias.

#![crate_name = "foo"]

use std::ops::Deref;

pub struct Foo<T>(T);

impl Foo<i32> {
    pub fn get_i32(&self) -> i32 { self.0 }
}

impl Foo<u32> {
    pub fn get_u32(&self) -> u32 { self.0 }
}

pub type X = Foo<i32>;

//@ has 'foo/struct.Bar.html'
// Only `get_i32` should be listed.
//@ count - '//*[@id="deref-methods-X-1"]//h4' 1
//@ has - '//*[@id="deref-methods-X-1"]//h4' 'get_i32'
// We check only `get_i32` is listed in the sidebar as well.
//@ count - '//*[@class="block deref-methods"]//a' 1
//@ has - '//*[@class="block deref-methods"]//a[@href="#method.get_i32"]' 'get_i32'
pub struct Bar;

impl Deref for Bar {
    type Target = X;
    fn deref(&self) -> &Self::Target {
        &X(0)
    }
}
