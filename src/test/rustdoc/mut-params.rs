// Rustdoc shouldn't display `mut` in function arguments, which are
// implementation details. Regression test for #81289.

#![crate_name = "foo"]

pub struct Foo;

// @!has foo/struct.Foo.html '//*[@class="impl-items"]//*[@class="method"]' 'mut'
impl Foo {
    pub fn foo(mut self) {}

    pub fn bar(mut bar: ()) {}
}

// @!has foo/fn.baz.html '//*[@class="rust fn"]' 'mut'
pub fn baz(mut foo: Foo) {}
