#![crate_name = "foo"]

// @has foo/fn.f.html
// @has - '//pre[@class="rust item-decl"]' 'pub fn f(callback: fn(len: usize, foo: u32))'
pub fn f(callback: fn(len: usize, foo: u32)) {}
