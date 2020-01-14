#![crate_name = "foo"]

// @has foo/fn.foo.html
// !@has - '//a[@href="http://a.a"]'
// @has - '//a[@href="#implementing-stuff-somewhere"]' 'Implementing stuff somewhere'

/// fooo
///
/// # Implementing [stuff](http://a.a) somewhere
///
/// hello
pub fn foo() {}
