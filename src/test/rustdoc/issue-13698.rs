// aux-build:issue-13698.rs
// ignore-cross-compile

extern crate issue_13698;

pub struct Foo;
// @!has issue_13698/struct.Foo.html '//*[@id="method.foo"]' 'fn foo'
// @!has - '//*[@id="method.foo2"]' 'fn foo2'
impl issue_13698::FooAux for Foo {
  fn foo2(&self) {}
}

pub trait Bar {
    #[doc(hidden)]
    fn bar(&self) {}
    #[doc(hidden)]
    fn qux(&self);
}

// @!has issue_13698/struct.Foo.html '//*[@id="method.bar"]' 'fn bar'
// @!has - '//*[@id="method.qux"]' 'fn qux'
impl Bar for Foo {
  fn qux(&self) {}
}
