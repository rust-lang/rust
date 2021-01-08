#![crate_name = "foo"]

// @has foo/fn.f.html
// @has - '//*[@class="rust fn"]' 'pub fn f(0u8 ...255: u8)'
pub fn f(0u8...255: u8) {}
