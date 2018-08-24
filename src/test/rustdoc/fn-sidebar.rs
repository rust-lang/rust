#![crate_name = "foo"]

// @has foo/fn.bar.html
// @has - '//*[@class="sidebar-elems"]' ''
pub fn bar() {}

// @has foo/constant.BAR.html
// @has - '//*[@class="sidebar-elems"]' ''
pub const BAR: u32 = 0;
