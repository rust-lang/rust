// Regression test for <https://github.com/rust-lang/rust/issues/110698>.
// This test ensures that the re-exported items still have the `#[repr(...)]` attribute.

// aux-build:repr.rs

#![crate_name = "foo"]

extern crate repr;

// @has 'foo/struct.ReprC.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(C, align(8))]'
pub use repr::ReprC;
// @has 'foo/struct.ReprSimd.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(simd, packed(2))]'
pub use repr::ReprSimd;
// @has 'foo/struct.ReprTransparent.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
pub use repr::ReprTransparent;
// @has 'foo/enum.ReprIsize.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(isize)]'
pub use repr::ReprIsize;
// @has 'foo/enum.ReprU8.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(u8)]'
pub use repr::ReprU8;

// Regression test for <https://github.com/rust-lang/rust/issues/90435>.
// Check that we show `#[repr(transparent)]` iff the non-1-ZST field is public or at least one
// field is public in case all fields are 1-ZST fields.

// @has 'foo/struct.ReprTransparentPrivField.html'
// @!has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
pub use repr::ReprTransparentPrivField;

// @has 'foo/struct.ReprTransparentPriv1ZstFields.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
pub use repr::ReprTransparentPriv1ZstFields;

// @has 'foo/struct.ReprTransparentPrivFieldPub1ZstFields.html'
// @!has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
pub use repr::ReprTransparentPrivFieldPub1ZstFields;
