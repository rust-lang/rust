// Test for the `macro_attr` and `macro_derive` features.

#![feature(macro_derive)]

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ count - '//*[@id="main-content"]/h2[@class="section-header"]' 1
//@ has - '//*[@id="main-content"]/h2[@class="section-header"]' 'Derive Macros'
//@ has - '//a[@href="derive.derive.html"]' 'derive'

//@ has 'foo/derive.derive.html'
//@ has - '//*[@class="rust item-decl"]/code' '#[derive(derive)]'

//@ has 'foo/all.html'
//@ count - '//*[@id="main-content"]/h3' 1
//@ has - '//*[@id="main-content"]/h3' 'Derive Macros'

#[macro_export]
macro_rules! derive {
    derive() () => {};
}
