#![crate_name = "foo"]

// @has foo/fn.f.html
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'pub fn f(_: u8)'
pub fn f(0u8..=255: u8) {}
