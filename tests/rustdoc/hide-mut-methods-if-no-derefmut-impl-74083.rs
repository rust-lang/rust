// https://github.com/rust-lang/rust/issues/74083
#![crate_name="foo"]

use std::ops::Deref;

pub struct Foo;

impl Foo {
    pub fn foo(&mut self) {}
}

//@ has foo/struct.Bar.html
//@ !has - '//div[@class="sidebar-links"]/a[@href="#method.foo"]' 'foo'
pub struct Bar {
    foo: Foo,
}

impl Deref for Bar {
    type Target = Foo;

    fn deref(&self) -> &Foo {
        &self.foo
    }
}
