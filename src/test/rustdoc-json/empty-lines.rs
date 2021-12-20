// @has empty_lines.json "$.index[*][?(@.name == 'foo')]"
// @has - "$.index[*][?(@.name == 'foo')].docs" \"\\n\\n\"
///
///
///
// ^ note that the above line does *not* include a trailing new line in the docs
pub fn foo() {}

// @has empty_lines.json "$.index[*][?(@.name == 'bar')].docs" "\"first line\\nsecond line \""
#[doc = "\n first line"]
#[doc = "\n second line \n"]
pub fn bar() {}
