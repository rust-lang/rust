use std::ops::Deref;

pub struct Foo;

impl Foo {
    pub fn foo(&mut self) {}
}

// @has issue_74083/struct.Bar.html
// @!has - '//div[@class="sidebar-links"]/a[@href="#method.foo"]' 'foo'
pub struct Bar {
    foo: Foo,
}

impl Deref for Bar {
    type Target = Foo;

    fn deref(&self) -> &Foo {
        &self.foo
    }
}
