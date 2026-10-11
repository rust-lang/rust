#![crate_name = "unicode"]

pub struct Foo;

impl Foo {
    //@ has unicode/type.Foo.html //a/@href "#%C3%BA"
    //@ !has unicode/type.Foo.html //a/@href "#ú"
    /// # ú
    pub fn foo() {}
}
