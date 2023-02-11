#![crate_name = "foo"]

// @has foo/fn.f.html
// @has - '//pre[@class="rust item-decl"]' 'pub fn f(_: u8)'
pub fn f(0u8..=255: u8) {}
