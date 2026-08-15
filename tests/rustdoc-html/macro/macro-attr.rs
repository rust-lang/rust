// Test for the `macro_attr` and `macro_derive` features.

#![feature(macro_attr)]

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ count - '//*[@id="main-content"]/h2[@class="section-header"]' 1
//@ has - '//*[@id="main-content"]/h2[@class="section-header"]' 'Attribute Macros'
//@ has - '//a[@href="attr.attr.html"]' 'attr'

//@ has 'foo/attr.attr.html'
//@ has - '//*[@class="rust item-decl"]/code' '#[attr]'

//@ has 'foo/all.html'
//@ count - '//*[@id="main-content"]/h3' 1
//@ has - '//*[@id="main-content"]/h3' 'Attribute Macros'

#[macro_export]
macro_rules! attr {
    attr() () => {};
}
