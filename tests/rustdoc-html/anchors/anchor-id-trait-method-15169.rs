//@ has issue_15169/type.Foo.html '//*[@id="method.eq"]' 'fn eq'

// https://github.com/rust-lang/rust/issues/15169
#![crate_name="issue_15169"]

#[derive(PartialEq)]
pub struct Foo;
