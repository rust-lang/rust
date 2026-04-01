//@ compile-flags: -Zunstable-options --html-no-source

// This test ensures that the `--html-no-source` flag disables
// the creation of the `src` folder.

#![feature(staged_api)]
#![stable(feature = "bar", since = "1.0")]
#![crate_name = "foo"]

// Ensures that there is no items in the corresponding "src" folder.
//@ files 'src/foo' '[]'

//@ has foo/fn.foo.html
//@ has - '//div[@class="main-heading"]/*[@class="sub-heading"]' '1.0.0'
//@ !has - '//div[@class="main-heading"]/*[@class="sub-heading"]' '1.0.0 · source'
#[stable(feature = "bar", since = "1.0")]
pub fn foo() {}

//@ has foo/struct.Bar.html
//@ has - '//div[@class="main-heading"]/*[@class="sub-heading"]' '1.0.0'
//@ !has - '//div[@class="main-heading"]/*[@class="sub-heading"]' '1.0.0 · source'
#[stable(feature = "bar", since = "1.0")]
pub struct Bar;

impl Bar {
    //@ has - '//*[@id="method.bar"]/*[@class="since rightside"]' '2.0.0'
    //@ !has - '//*[@id="method.bar"]/*[@class="rightside"]' '2.0.0 ·'
    #[stable(feature = "foobar", since = "2.0")]
    pub fn bar() {}
}
