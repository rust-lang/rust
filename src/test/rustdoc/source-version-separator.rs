#![stable(feature = "bar", since = "1.0")]
#![crate_name = "foo"]
#![feature(staged_api)]

// @has foo/trait.Bar.html
// @has - '//div[@class="main-heading"]/*[@class="out-of-band"]' '1.0 · source · '
#[stable(feature = "bar", since = "1.0")]
pub trait Bar {
    // @has - '//*[@id="tymethod.foo"]/*[@class="rightside"]' '3.0 · source'
    #[stable(feature = "foobar", since = "3.0")]
    fn foo();
}

// @has - '//div[@id="implementors-list"]//*[@class="rightside"]' '4.0 · source'

// @has foo/struct.Foo.html
// @has - '//div[@class="main-heading"]/*[@class="out-of-band"]' '1.0 · source · '
#[stable(feature = "baz", since = "1.0")]
pub struct Foo;

impl Foo {
    // @has - '//*[@id="method.foofoo"]/*[@class="rightside"]' '3.0 · source'
    #[stable(feature = "foobar", since = "3.0")]
    pub fn foofoo() {}
}

#[stable(feature = "yolo", since = "4.0")]
impl Bar for Foo {
    fn foo() {}
}
