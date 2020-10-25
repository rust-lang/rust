#![crate_name = "foo"]

// @has foo/fn.foo.html
// @has - '//ul/li/ul/li/p' 'A sublist.'
// @has - '//ul/li/ul/li/p' 'Another sublist.'
/// Some text.
///
/// * A list.
///
///    * A sublist.
///
///    * Another sublist.
pub fn foo() {}
