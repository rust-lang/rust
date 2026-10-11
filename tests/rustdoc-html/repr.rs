// Test the rendering of `#[repr]` on ADTs.
#![feature(repr_simd)] // only used for the `ReprSimd` test case
#![feature(rustc_attrs)] // only used for the `RustcPubTransparent` test case

// Check the "local case" (HIR cleaning) //

// Don't render the default repr which is `Rust`.
//@ has 'repr/type.ReprDefault.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(Rust)]'
pub struct ReprDefault;

// Don't render the `Rust` repr — even if given explicitly — since it's the default.
//@ has 'repr/type.ReprRust.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(Rust)]'
#[repr(Rust)] // omitted
pub struct ReprRust;

//@ has 'repr/type.ReprCPubFields.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(C)]'
#[repr(C)] // public
pub struct ReprCPubFields {
    pub a: u32,
    pub b: u32,
}

//@ has 'repr/type.ReprCPrivField.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(C)]'
#[repr(C)] // private...
pub struct ReprCPrivField {
    a: u32, // ...since this is private
    pub b: u32,
}

//@ has 'repr/type.ReprIsize.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(isize)]'
#[repr(isize)] // public
pub enum ReprIsize {
    Bla,
}

//@ has 'repr/type.ReprU32HiddenVariant.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(u32)]'
#[repr(u32)] // private...
pub enum ReprU32HiddenVariant {
    #[doc(hidden)]
    Hidden, // ...since this is hidden
    Public,
}

//@ has 'repr/type.ReprAlignHiddenField.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(align(4))]'
#[repr(align(4))] // private...
pub struct ReprAlignHiddenField {
    #[doc(hidden)]
    pub hidden: i16, // ...since this field is hidden
}

//@ has 'repr/type.ReprSimd.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(simd, packed(2))]'
#[repr(simd, packed(2))] // public
pub struct ReprSimd {
    pub field: [u8; 1],
}

//@ has 'repr/type.ReprU32Align.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(u32, align(8))]'
#[repr(u32, align(8))] // public
pub enum ReprU32Align {
    Variant(u16),
}

//@ has 'repr/type.ReprCHiddenVariantField.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(C)]'
#[repr(C)] // private...
pub enum ReprCHiddenVariantField {
    Variant { #[doc(hidden)] field: () }, //...since this field is hidden
}

//@ has 'repr/type.ReprTransparentPrivField.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // private...
pub struct ReprTransparentPrivField {
    field: u32, // ...since the non-1-ZST field is private
}

//@ has 'repr/type.ReprTransparentPriv1ZstFields.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // public...
pub struct ReprTransparentPriv1ZstFields {
    marker0: Marker,
    pub main: u64, // ...since the non-1-ZST field is public and visible
    marker1: Marker,
} // the two private 1-ZST fields don't matter

//@ has 'repr/type.ReprTransparentPrivFieldPub1ZstField.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // private...
pub struct ReprTransparentPrivFieldPub1ZstField {
    main: [u16; 0], // ...since the non-1-ZST field is private
    pub marker: Marker, // this public 1-ZST field doesn't matter
}

//@ has 'repr/type.ReprTransparentPub1ZstField.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // public...
pub struct ReprTransparentPub1ZstField {
    marker0: Marker, // ...since we don't have a non-1-ZST field...
    pub marker1: Marker, // ...and this field is public and visible
}

//@ has 'repr/type.ReprTransparentUnitStruct.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // public
pub struct ReprTransparentUnitStruct;

//@ has 'repr/type.ReprTransparentEnumUnitVariant.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // public
pub enum ReprTransparentEnumUnitVariant {
    Variant,
}

//@ has 'repr/type.ReprTransparentEnumHiddenUnitVariant.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // private
pub enum ReprTransparentEnumHiddenUnitVariant {
    #[doc(hidden)] Variant(u32),
}

//@ has 'repr/type.ReprTransparentEnumPub1ZstField.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // public...
pub enum ReprTransparentEnumPub1ZstField {
    Variant {
        field: u64, // ...since the non-1-ZST field is public
        #[doc(hidden)]
        marker: Marker, // this hidden 1-ZST field doesn't matter
    },
}

//@ has 'repr/type.ReprTransparentEnumHidden1ZstField.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // private...
pub enum ReprTransparentEnumHidden1ZstField {
    Variant {
        #[doc(hidden)]
        field: u64, // ...since the non-1-ZST field is public
    },
}

// Regression test for <https://github.com/rust-lang/rust/issues/150919>.
//@ has 'repr/type.RustcPubTransparent.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // public...
#[rustc_pub_transparent] // ...since this attribute was applied
pub struct RustcPubTransparent {
    // ...despite this field being private
    field: u64,
}

struct Marker; // 1-ZST

// Check the "extern case" (middle cleaning) //

// Internally, HIR and middle cleaning share `#[repr]` rendering.
// Thus we'll only test the very basics in this section.

//@ aux-build: ext-repr.rs
extern crate ext_repr as ext;

// Regression test for <https://github.com/rust-lang/rust/issues/110698>.
//@ has 'repr/type.ReprI8.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(i8)]'
pub use ext::ReprI8;
