// compile-flags:-Z unstable-options --generate-redirect-pages

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

// @has issue_19190/Bar.t.html
// @has issue_19190/struct.Bar.html
// @has - '//*[@id="foo.v"]' 'fn foo(&self)'
// @has - '//*[@id="method.foo"]' 'fn foo(&self)'
// @!has - '//*[@id="static_foo.v"]' 'fn static_foo()'
// @!has - '//*[@id="method.static_foo"]' 'fn static_foo()'
