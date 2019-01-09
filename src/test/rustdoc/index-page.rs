// compile-flags: -Z unstable-options --enable-index-page

#![crate_name = "foo"]

// @has foo/../index.html
// @has - '//span[@class="in-band"]' 'List of all crates'
// @has - '//ul[@class="mod"]//a[@href="foo/index.html"]' 'foo'
pub struct Foo;
