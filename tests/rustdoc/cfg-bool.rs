#![feature(doc_cfg)]
#![crate_name = "foo"]

// regression test for https://github.com/rust-lang/rust/issues/138112

//@ has 'foo/fn.foo.html' '//div[@class="stab portability"]' 'Available nowhere'
#[doc(cfg(false))]
pub fn foo() {}

// a cfg(true) will simply be ommited, as it is the same as no cfg.
//@ !has 'foo/fn.bar.html' '//div[@class="stab portability"]' ''
#[doc(cfg(true))]
pub fn bar() {}
