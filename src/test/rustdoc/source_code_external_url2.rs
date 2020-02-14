// ignore-tidy-linelength
// compile-flags: -Z unstable-options --source-code-external-url https://a.a/

#![crate_name = "foo"]

// @has foo/struct.Foo.html
// @has - '//h1[@class="fqn"]//a[@href="https://a.a/foo/source_code_external_url2.rs.html#8"]' '[src]'
pub struct Foo;
