#![feature(doc_auto_cfg)]
#![crate_name = "foo"]

// @has foo/fn.foo.html
// @has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-doctest'
#[cfg(not(doctest))]
pub fn foo() {}

// @has foo/fn.bar.html
// @has - '//*[@class="item-info"]/*[@class="stab portability"]' 'doc'
// @!has - '//*[@class="item-info"]/*[@class="stab portability"]' 'test'
#[cfg(any(test, doc))]
pub fn bar() {}

// @has foo/fn.appear_1.html
// @has - '//*[@class="item-info"]/*[@class="stab portability"]' 'doc'
// @!has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-test'
#[cfg(any(doc, not(test)))]
pub fn appear_1() {} // issue #98065

// @has foo/fn.appear_2.html
// @has - '//*[@class="item-info"]/*[@class="stab portability"]' 'doc'
// @!has - '//*[@class="item-info"]/*[@class="stab portability"]' 'test'
#[cfg(any(doc, all(test)))]
pub fn appear_2() {} // issue #98065

// @has foo/fn.appear_3.html
// @has - '//*[@class="item-info"]/*[@class="stab portability"]' 'doc'
#[cfg(any(doc, all()))]
pub fn appear_3() {} // issue #98065
