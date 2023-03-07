#![crate_name = "foo"]

// @has foo/fn.f.html
// @has - //del "Y"
/// ~~Y~~
pub fn f() {}
