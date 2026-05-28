#![crate_name = "unicode"]

pub struct Foo;

impl Foo {
    //@ has unicode/struct.Foo.html //a/@href "#%C3%BA"
    //@ !has unicode/struct.Foo.html //a/@href "#ú"
    /// # ú
    pub fn foo() {}
}
