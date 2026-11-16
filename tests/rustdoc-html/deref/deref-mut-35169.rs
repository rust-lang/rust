// https://github.com/rust-lang/rust/issues/35169
#![crate_name="foo"]

use std::ops::Deref;

pub struct Foo;
pub struct Bar;

impl Foo {
    pub fn by_ref(&self) {}
    pub fn by_explicit_ref(self: &Foo) {}
    pub fn by_mut_ref(&mut self) {}
    pub fn by_explicit_mut_ref(self: &mut Foo) {}
    pub fn by_explicit_box(self: Box<Foo>) {}
    pub fn by_explicit_self_box(self: Box<Self>) {}
    pub fn static_foo() {}
}

impl Deref for Bar {
    type Target = Foo;
    fn deref(&self) -> &Foo { loop {} }
}

//@ has foo/struct.Bar.html
//@ has - '//*[@id="method.by_ref"]//h4[@class="code-header"]' 'fn by_ref(&self)'
//@ has - '//*[@id="method.by_ref"]' 'fn by_ref(&self)'
//@ has - '//*[@id="method.by_explicit_ref"]//h4[@class="code-header"]' 'fn by_explicit_ref(self: &Foo)'
//@ has - '//*[@id="method.by_explicit_ref"]' 'fn by_explicit_ref(self: &Foo)'
//@ !has - '//*[@id="method.by_mut_ref"]//h4[@class="code-header"]' 'fn by_mut_ref(&mut self)'
//@ !has - '//*[@id="method.by_mut_ref"]' 'fn by_mut_ref(&mut self)'
//@ !has - '//*[@id="method.by_explicit_mut_ref"]//h4[@class="code-header"]' 'fn by_explicit_mut_ref(self: &mut Foo)'
//@ !has - '//*[@id="method.by_explicit_mut_ref"]' 'fn by_explicit_mut_ref(self: &mut Foo)'
//@ !has - '//*[@id="method.by_explicit_box"]//h4[@class="code-header"]' 'fn by_explicit_box(self: Box<Foo>)'
//@ !has - '//*[@id="method.by_explicit_box"]' 'fn by_explicit_box(self: Box<Foo>)'
//@ !has - '//*[@id="method.by_explicit_self_box"]//h4[@class="code-header"]' 'fn by_explicit_self_box(self: Box<Self>)'
//@ !has - '//*[@id="method.by_explicit_self_box"]' 'fn by_explicit_self_box(self: Box<Self>)'
//@ !has - '//*[@id="method.static_foo"]//h4[@class="code-header"]' 'fn static_foo()'
//@ !has - '//*[@id="method.static_foo"]' 'fn static_foo()'
