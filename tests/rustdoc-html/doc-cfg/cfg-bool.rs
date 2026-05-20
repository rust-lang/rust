#![feature(doc_cfg)]
#![crate_name = "foo"]

// regression test for https://github.com/rust-lang/rust/issues/138112

//@ has 'foo/index.html'
//@ has - '//*[@class="stab portability"]/@title' 'Available nowhere'

//@ count 'foo/fn.foo.html' '//*[@class="stab portability"]' 1
//@ has 'foo/fn.foo.html' '//*[@class="stab portability"]' 'Available nowhere'
#[doc(cfg(false))]
pub fn foo() {}

// a cfg(true) will simply be omitted, as it is the same as no cfg.
//@ count 'foo/fn.bar.html' '//*[@class="stab portability"]' 0
#[doc(cfg(true))]
pub fn bar() {}
