// https://github.com/rust-lang/rust/issues/19190
#![crate_name="issue_19190"]

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

//@ has issue_19190/struct.Bar.html
//@ has - '//*[@id="method.foo"]//h4[@class="code-header"]' 'fn foo(&self)'
//@ has - '//*[@id="method.foo"]' 'fn foo(&self)'
//@ !has - '//*[@id="method.static_foo"]//h4[@class="code-header"]' 'fn static_foo()'
//@ !has - '//*[@id="method.static_foo"]' 'fn static_foo()'
