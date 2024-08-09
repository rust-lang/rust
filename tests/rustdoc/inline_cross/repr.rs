// Regression test for <https://github.com/rust-lang/rust/issues/110698>.
// This test ensures that the re-exported items still have the `#[repr(...)]` attribute.

//@ aux-build:repr.rs

#![crate_name = "foo"]

extern crate repr;

// Never display `repr(Rust)` since it's the default anyway.
//@ has 'foo/struct.ReprRust.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(Rust)]'
pub use repr::ReprRust;

//@ has 'foo/struct.ReprC.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(C, align(8))]'
pub use repr::ReprC;

//@ has 'foo/struct.ReprCPrivField.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(C)]'
pub use repr::ReprCPrivField;

//@ has 'foo/struct.ReprAlignHiddenField.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(align(4))]'
pub use repr::ReprAlignHiddenField;

//@ has 'foo/struct.ReprSimd.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(simd, packed(2))]'
pub use repr::ReprSimd;

//@ has 'foo/struct.ReprTransparent.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
pub use repr::ReprTransparent;

//@ has 'foo/enum.ReprIsize.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(isize)]'
pub use repr::ReprIsize;

//@ has 'foo/enum.ReprU8.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(u8)]'
pub use repr::ReprU8;

//@ has 'foo/enum.ReprU32.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(u32)]'
pub use repr::ReprU32;

//@ has 'foo/enum.ReprU64HiddenVariants.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(u64)]'
pub use repr::ReprU64HiddenVariants;

// Regression test for <https://github.com/rust-lang/rust/issues/90435>.
// Check that we show `#[repr(transparent)]` iff the non-1-ZST field is public or at least one
// field is public in case all fields are 1-ZST fields.

//@ has 'foo/struct.ReprTransparentPrivField.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
pub use repr::ReprTransparentPrivField;

//@ has 'foo/struct.ReprTransparentPriv1ZstFields.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
pub use repr::ReprTransparentPriv1ZstFields;

//@ has 'foo/struct.ReprTransparentPrivFieldPub1ZstFields.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
pub use repr::ReprTransparentPrivFieldPub1ZstFields;
