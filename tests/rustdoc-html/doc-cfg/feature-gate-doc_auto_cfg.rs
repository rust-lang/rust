#![feature(doc_cfg)]

#![crate_name = "foo"]

//@ has foo/fn.foo.html
//@ count - '//*[@class="item-info"]/*[@class="stab portability"]' 0
#[cfg(not(test))]
pub fn foo() {}
