// Test for the `macro_attr` and `macro_derive` features.

#![feature(macro_attr)]
#![feature(macro_derive)]

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ count - '//*[@id="main-content"]/h2[@class="section-header"]' 3
//@ has - '//*[@id="main-content"]/h2[@class="section-header"]' 'Attribute Macros'
//@ has - '//*[@id="main-content"]/h2[@class="section-header"]' 'Derive Macros'
//@ has - '//*[@id="main-content"]/h2[@class="section-header"]' 'Macros'
//@ has - '//a[@href="macro.all.html"]' 'all'

//@ has 'foo/macro.all.html'
//@ has - '//*[@class="macro-info"]' 'ⓘ This is an attribute/derive/function macro'

//@ has 'foo/all.html'
//@ count - '//*[@id="main-content"]/h3' 3
//@ has - '//*[@id="main-content"]/h3' 'Attribute Macros'
//@ has - '//*[@id="main-content"]/h3' 'Derive Macros'
//@ has - '//*[@id="main-content"]/h3' 'Macros'

#[macro_export]
macro_rules! all {
    () => {};
    attr() () => {};
    derive() () => {};
}
