// @has issue_15169/struct.Foo.html '//*[@id="method.eq"]' 'fn eq'
#[derive(PartialEq)]
pub struct Foo;
