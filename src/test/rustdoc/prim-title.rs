#![crate_name = "foo"]

// @has foo/primitive.u8.html '//head/title' 'u8 - Rust'
// @!has - '//head/title' 'foo'
#[doc(primitive = "u8")]
/// `u8` docs
mod u8 {}
