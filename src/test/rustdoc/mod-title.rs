#![crate_name = "foo"]

// @has foo/bar/index.html '//head/title' 'foo::bar - Rust'
/// blah
pub mod bar {
    pub fn a() {}
}

// @has foo/baz/index.html '//head/title' 'foo::baz - Rust'
pub mod baz {
    pub fn a() {}
}
