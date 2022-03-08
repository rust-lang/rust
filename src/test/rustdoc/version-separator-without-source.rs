#![doc(html_no_source)]
#![feature(staged_api)]
#![stable(feature = "bar", since = "1.0")]
#![crate_name = "foo"]

// @has foo/fn.foo.html
// @has - '//div[@class="main-heading"]/*[@class="out-of-band"]' '1.0 · '
// @!has - '//div[@class="main-heading"]/*[@class="out-of-band"]' '1.0 · source · '
#[stable(feature = "bar", since = "1.0")]
pub fn foo() {}

// @has foo/struct.Bar.html
// @has - '//div[@class="main-heading"]/*[@class="out-of-band"]' '1.0 · '
// @!has - '//div[@class="main-heading"]/*[@class="out-of-band"]' '1.0 · source · '
#[stable(feature = "bar", since = "1.0")]
pub struct Bar;

impl Bar {
    // @has - '//*[@id="method.bar"]/*[@class="rightside"]' '2.0'
    // @!has - '//*[@id="method.bar"]/*[@class="rightside"]' '2.0 ·'
    #[stable(feature = "foobar", since = "2.0")]
    pub fn bar() {}
}
