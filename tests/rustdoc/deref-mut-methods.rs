#![crate_name = "foo"]

use std::ops;

pub struct Foo;

impl Foo {
    pub fn foo(&mut self) {}
}

// @has foo/struct.Bar.html
// @has - '//*[@class="sidebar-elems"]//*[@class="block"]//a[@href="#method.foo"]' 'foo'
pub struct Bar {
    foo: Foo,
}

impl ops::Deref for Bar {
    type Target = Foo;

    fn deref(&self) -> &Foo {
        &self.foo
    }
}

impl ops::DerefMut for Bar {
    fn deref_mut(&mut self) -> &mut Foo {
        &mut self.foo
    }
}
