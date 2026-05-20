// This test ensures that `auto_cfg(hide)` on a key also hides `key = value`.

#![feature(doc_cfg)]
#![crate_name = "foo"]

#![doc(auto_cfg(hide(meow)))]

//@ has foo/fn.foo.html
//@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob'
//@ !has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow'
#[cfg(not(meow))]
#[cfg(not(blob))]
pub fn foo() {}

//@ has foo/fn.bar.html
//@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob=lol'
//@ !has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow=lol'
#[cfg(not(meow = "lol"))]
#[cfg(not(blob = "lol"))]
pub fn bar() {}
