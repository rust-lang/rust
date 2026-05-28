// Test for `custom_code_classes_in_docs` feature.

#![crate_name = "foo"]
#![feature(no_core)]
#![no_core]

//@ has 'foo/struct.Bar.html'
//@ has - '//*[@id="main-content"]//pre[@class="language-whatever hoho-c"]' 'main;'
//@ has - '//*[@id="main-content"]//pre[@class="language-whatever2 haha-c"]' 'main;'
//@ has - '//*[@id="main-content"]//pre[@class="language-whatever4 huhu-c"]' 'main;'

/// ```{class=hoho-c},whatever
/// main;
/// ```
///
/// Testing multiple kinds of orders.
///
/// ```whatever2 {class=haha-c}
/// main;
/// ```
///
/// Testing with multiple "unknown". Only the first should be used.
///
/// ```whatever4,{.huhu-c} whatever5
/// main;
/// ```
pub struct Bar;
