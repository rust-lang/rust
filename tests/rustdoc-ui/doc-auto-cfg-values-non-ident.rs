// Regression test for https://github.com/rust-lang/rust/issues/158744.

#![feature(doc_cfg)]

#[doc(auto_cfg(hide(a, values(::b))))]
//~^ ERROR malformed `doc` attribute input
fn f() {}
