//@ aux-build:issue-17476.rs
//@ ignore-cross-compile
// https://github.com/rust-lang/rust/issues/17476

#![crate_name="issue_17476"]

extern crate issue_17476;

pub struct Foo;

//@ has issue_17476/struct.Foo.html \
//      '//*[@href="http://example.com/issue_17476/trait.Foo.html#method.foo"]' \
//      'foo'
impl issue_17476::Foo for Foo {}
