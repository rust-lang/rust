use std::ops::Deref;

pub struct Foo;
pub struct Bar;

impl Foo {
    pub fn foo(&self) {}
    pub fn static_foo() {}
}

impl Deref for Bar {
    type Target = Foo;
    fn deref(&self) -> &Foo { loop {} }
}

// @has issue_19190/struct.Bar.html
// @has - '//*[@id="method.foo"]//code' 'fn foo(&self)'
// @has - '//*[@id="method.foo"]' 'fn foo(&self)'
// @!has - '//*[@id="method.static_foo"]//code' 'fn static_foo()'
// @!has - '//*[@id="method.static_foo"]' 'fn static_foo()'
