#![crate_name = "foo"]

// @has static/8a40d4987fbb905
// @has foo/struct.Enum.html
// @has - '//img[@src="../static/8a40d4987fbb905"]' ''

/// Image test!
///
/// ![osef](src/test/rustdoc/copy-local-img.rs)
pub struct Enum;
