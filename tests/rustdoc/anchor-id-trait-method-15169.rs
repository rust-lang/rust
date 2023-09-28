// @has issue_15169/struct.Foo.html '//*[@id="method.eq"]' 'fn eq'

#![crate_name="issue_15169"]

#[derive(PartialEq)]
pub struct Foo;
