#![feature(doc_auto_cfg)]

#![crate_name = "foo"]

// @has foo/fn.foo.html
// @has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-test'
#[cfg(not(test))]
pub fn foo() {}
