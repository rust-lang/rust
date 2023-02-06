// Test ensuring that local resources copy is working as expected.
// Original issue: <https://github.com/rust-lang/rust/issues/32104>

#![crate_name = "foo"]
#![feature(no_core)]
#![no_std]
#![no_core]

// @has local_resources/foo/0.svg
// @has foo/struct.Enum.html
// @has - '//img[@src="../local_resources/foo/0.svg"]' ''
/// test!
///
/// ![yep](../../src/librustdoc/html/static/images/rust-logo.svg)
pub struct Enum;

pub mod sub {
    // @has foo/sub/struct.Enum.html
    // @has - '//img[@src="../../local_resources/foo/0.svg"]' ''
    /// test!
    ///
    /// ![yep](../../src/librustdoc/html/static/images/rust-logo.svg)
    pub struct Enum;
}
