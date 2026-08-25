// Rustdoc shouldn't display `mut` in function arguments, which are
// implementation details. Regression test for #81289.

#![crate_name = "foo"]

pub struct Foo;

//@ count foo/struct.Foo.html '//*[@class="impl-items"]//*[@class="method"]' 2
//@ !has - '//*[@class="impl-items"]//*[@class="method"]' 'mut'
impl Foo {
    pub fn foo(mut self) {}

    pub fn bar(mut bar: ()) {}
}

//@ count foo/fn.baz.html '//pre[@class="rust item-decl"]' 1
//@ !has - '//pre[@class="rust item-decl"]' 'mut'
pub fn baz(mut foo: Foo) {}
