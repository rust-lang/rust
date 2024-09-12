// ignore-tidy-linelength
// https://github.com/rust-lang/rust/issues/100679
#![crate_name="foo"]

//@ has foo/impls/struct.Foo.html
//@ has - '//h3[@class="code-header"]' 'impl Foo'
//@ has - '//h3[@class="code-header"]' 'impl Bar for Foo'
//@ count - '//*[@id="trait-implementations-list"]//*[@class="impl"]' 1
//@ count - '//*[@id="main-content"]/div[@id="implementations-list"]/details/summary/*[@class="impl"]' 1
//@ has foo/impls/bar/trait.Bar.html
//@ has - '//h3[@class="code-header"]' 'impl Bar for Foo'
//@ count - '//*[@class="struct"]' 1
pub mod impls;

#[doc(inline)]
pub use impls as impls2;
