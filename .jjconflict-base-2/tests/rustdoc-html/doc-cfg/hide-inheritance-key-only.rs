// This test ensures that `auto_cfg(hide(key))` does not hide `key = value` and that
// `auto_cfg(hide(key, values(none())))` does the same.

#![feature(doc_cfg)]
#![crate_name = "foo"]

#![doc(auto_cfg(hide(meow)))]
#![doc(auto_cfg(hide(another_meow, values(none()))))]

//@ has foo/fn.foo.html
//@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob'
//@ !has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow'
//@ !has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-another_meow'
#[cfg(not(meow))]
#[cfg(not(another_meow))]
#[cfg(not(blob))]
pub fn foo() {}

//@ has foo/fn.bar.html
//@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob=lol'
//@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow=lol'
//@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-another_meow=lol'
#[cfg(not(meow = "lol"))]
#[cfg(not(another_meow = "lol"))]
#[cfg(not(blob = "lol"))]
pub fn bar() {}
