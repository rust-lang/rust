use std::ops::Deref;
use std::ops::DerefMut;

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

impl DerefMut for Bar {
    fn deref_mut(&mut self) -> &mut Foo { loop {} }
}

// @has issue_35169_2/struct.Bar.html
// @has - '//*[@id="by_ref.v"]' 'fn by_ref(&self)'
// @has - '//*[@id="method.by_ref"]' 'fn by_ref(&self)'
// @has - '//*[@id="by_explicit_ref.v"]' 'fn by_explicit_ref(self: &Foo)'
// @has - '//*[@id="method.by_explicit_ref"]' 'fn by_explicit_ref(self: &Foo)'
// @has - '//*[@id="by_mut_ref.v"]' 'fn by_mut_ref(&mut self)'
// @has - '//*[@id="method.by_mut_ref"]' 'fn by_mut_ref(&mut self)'
// @has - '//*[@id="by_explicit_mut_ref.v"]' 'fn by_explicit_mut_ref(self: &mut Foo)'
// @has - '//*[@id="method.by_explicit_mut_ref"]' 'fn by_explicit_mut_ref(self: &mut Foo)'
// @!has - '//*[@id="by_explicit_box.v"]' 'fn by_explicit_box(self: Box<Foo>)'
// @!has - '//*[@id="method.by_explicit_box"]' 'fn by_explicit_box(self: Box<Foo>)'
// @!has - '//*[@id="by_explicit_self_box.v"]' 'fn by_explicit_self_box(self: Box<Self>)'
// @!has - '//*[@id="method.by_explicit_self_box"]' 'fn by_explicit_self_box(self: Box<Self>)'
// @!has - '//*[@id="static_foo.v"]' 'fn static_foo()'
// @!has - '//*[@id="method.static_foo"]' 'fn static_foo()'
