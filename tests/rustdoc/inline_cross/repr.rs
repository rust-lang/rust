// Regression test for <https://github.com/rust-lang/rust/issues/110698>.
// This test ensures that the re-exported items still have the `#[repr(...)]` attribute.

// aux-build:repr.rs

#![crate_name = "foo"]

extern crate repr;

// @has 'foo/struct.ReprC.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(C, align(8))]'
#[doc(inline)]
pub use repr::ReprC;
// @has 'foo/struct.ReprSimd.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(simd, packed(2))]'
#[doc(inline)]
pub use repr::ReprSimd;
// @has 'foo/struct.ReprTransparent.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[doc(inline)]
pub use repr::ReprTransparent;
// @has 'foo/enum.ReprIsize.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(isize)]'
#[doc(inline)]
pub use repr::ReprIsize;
// @has 'foo/enum.ReprU8.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(u8)]'
#[doc(inline)]
pub use repr::ReprU8;
