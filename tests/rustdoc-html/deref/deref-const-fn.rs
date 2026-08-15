// This test ensures that the const methods from Deref aren't shown as const.
// For more information, see https://github.com/rust-lang/rust/issues/90855.

#![crate_name = "foo"]

#![feature(staged_api)]

#![stable(feature = "rust1", since = "1.0.0")]

//@ has 'foo/struct.Bar.html'
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Bar;

impl Bar {
    //@ has - '//*[@id="method.len"]' 'pub const fn len(&self) -> usize'
    //@ has - '//*[@id="method.len"]//span[@class="since"]' 'const: 1.0.0'
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "rust1", since = "1.0.0")]
    pub const fn len(&self) -> usize { 0 }
}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Foo {
    value: Bar,
}

//@ has 'foo/struct.Foo.html'
//@ has - '//*[@id="method.len"]' 'pub fn len(&self) -> usize'
//@ has - '//*[@id="method.len"]//span[@class="since"]' '1.0.0'
//@ !has - '//*[@id="method.len"]//span[@class="since"]' '(const: 1.0.0)'
#[stable(feature = "rust1", since = "1.0.0")]
impl std::ops::Deref for Foo {
    type Target = Bar;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
