// This is a regression test for <https://github.com/rust-lang/rust/issues/118195>.
// It ensures that the "Fields" heading is not generated if no field is displayed.

#![crate_name = "foo"]

//@ has 'foo/enum.Foo.html'
//@ has - '//*[@id="variant.A"]' 'A'
//@ count - '//*[@id="variant.A.fields"]' 0
//@ has - '//*[@id="variant.B"]' 'B'
//@ count - '//*[@id="variant.B.fields"]' 0
//@ snapshot variants - '//*[@id="main-content"]/*[@class="variants"]'

pub enum Foo {
    /// A variant with no fields
    A {},
    /// A variant with hidden fields
    B { #[doc(hidden)] a: u8 },
}
