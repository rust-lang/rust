//@ aux-build:issue-13698.rs
//@ ignore-cross-compile

// https://github.com/rust-lang/rust/issues/13698
#![crate_name="issue_13698"]

extern crate issue_13698;

pub struct Foo;
//@ !has issue_13698/struct.Foo.html '//*[@id="method.foo"]' 'fn foo'
impl issue_13698::Foo for Foo {}

pub trait Bar {
    #[doc(hidden)]
    fn bar(&self) {}
}

//@ !has issue_13698/struct.Foo.html '//*[@id="method.bar"]' 'fn bar'
impl Bar for Foo {}
