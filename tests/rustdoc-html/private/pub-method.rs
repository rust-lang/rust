//@ compile-flags: --document-private-items

#![crate_name = "foo"]

//@ has foo/fn.bar.html
//@ has - '//pre[@class="rust item-decl"]' 'pub fn bar() -> '
/// foo
pub fn bar() -> usize {
    2
}

//@ has foo/struct.Foo.html
//@ has - '//*[@class="method"]' 'pub fn new()'
//@ has - '//*[@class="method"]' 'fn not_pub()'
pub struct Foo(usize);

impl Foo {
    pub fn new() -> Foo { Foo(0) }
    fn not_pub() {}
}
