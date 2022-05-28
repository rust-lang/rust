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
