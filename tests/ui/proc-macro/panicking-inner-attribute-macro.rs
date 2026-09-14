//! Regression test for <https://github.com/rust-lang/rust/issues/99478>.

//@ proc-macro: panicking-attribute.rs
//@ compile-flags: --crate-type=lib

#![feature(custom_inner_attributes)]
#![panicking_attribute::tester]
//~^ ERROR custom attribute panicked
