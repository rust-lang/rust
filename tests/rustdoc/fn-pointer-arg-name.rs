#![crate_name = "foo"]

// @has foo/fn.f.html
// @has - '//div[@class="item-decl"]/pre[@class="rust"]' 'pub fn f(callback: fn(len: usize, foo: u32))'
pub fn f(callback: fn(len: usize, foo: u32)) {}
