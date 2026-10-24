// Test for the `macro_attr` and `macro_derive` features.

#![feature(macro_attr)]
#![feature(macro_derive)]

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ count - '//*[@id="main-content"]/h2[@class="section-header"]' 2
//@ has - '//*[@id="main-content"]/h2[@class="section-header"]' 'Attribute Macros'
//@ has - '//*[@id="main-content"]/h2[@class="section-header"]' 'Derive Macros'
//@ has - '//a[@href="macro.no_bang.html"]' 'no_bang'

//@ has 'foo/macro.no_bang.html'
//@ has - '//*[@class="macro-info"]' 'ⓘ This is an attribute/derive macro'

//@ has 'foo/all.html'
//@ count - '//*[@id="main-content"]/h3' 2
//@ has - '//*[@id="main-content"]/h3' 'Attribute Macros'
//@ has - '//*[@id="main-content"]/h3' 'Derive Macros'

#[macro_export]
macro_rules! no_bang {
    attr() () => {};
    derive() () => {};
}
