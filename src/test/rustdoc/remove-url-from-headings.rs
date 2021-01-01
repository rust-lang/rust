#![crate_name = "foo"]

// @has foo/fn.foo.html
// @!has - '//a[@href="http://a.a"]'
// @has - '//a[@href="#implementing-stuff-somewhere"]' 'Implementing stuff somewhere'
// @has - '//a[@href="#another-one-urg"]' 'Another one urg'

/// fooo
///
/// # Implementing [stuff](http://a.a "title") somewhere
///
/// hello
///
/// # Another [one][two] urg
///
/// [two]: http://a.a
pub fn foo() {}
