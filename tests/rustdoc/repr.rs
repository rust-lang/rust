// Never display `repr(Rust)` since it's the default anyway.
//@ has 'repr/struct.ReprRust.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(Rust)]'
#[repr(Rust)]
pub struct ReprRust;

//@ has 'repr/struct.ReprCPubFields.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(C)]'
#[repr(C)] // **public**
pub struct ReprCPubFields {
    pub a: u32,
    pub b: u32,
}

//@ has 'repr/struct.ReprCPrivFields.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(C)]'
#[repr(C)] // private
pub struct ReprCPrivFields {
    a: u32,
    b: u32,
}

//@ has 'repr/enum.ReprU32Align.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(u32, align(8))]'
#[repr(u32, align(8))] // **public**
pub enum ReprU32Align {
    Variant(u16),
}

// Regression test for <https://github.com/rust-lang/rust/issues/90435>.
// Check that we show `#[repr(transparent)]` iff the non-1-ZST field is public or at least one
// field is public in case all fields are 1-ZST fields.

//@ has 'repr/struct.ReprTransparentPrivField.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // private
pub struct ReprTransparentPrivField {
    field: u32, // non-1-ZST field
}

//@ has 'repr/struct.ReprTransparentPriv1ZstFields.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // **public**
pub struct ReprTransparentPriv1ZstFields {
    marker0: Marker,
    pub main: u64, // non-1-ZST field
    marker1: Marker,
}

//@ has 'repr/struct.ReprTransparentPub1ZstField.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // **public**
pub struct ReprTransparentPub1ZstField {
    marker0: Marker,
    pub marker1: Marker,
}

//@ has 'repr/struct.ReprTransparentUnitStruct.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // **public**
pub struct ReprTransparentUnitStruct;

//@ has 'repr/enum.ReprTransparentEnumUnitVariant.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // **public**
pub enum ReprTransparentEnumUnitVariant {
    Variant,
}

//@ has 'repr/enum.ReprTransparentEnumHiddenUnitVariant.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // private
pub enum ReprTransparentEnumHiddenUnitVariant {
    #[doc(hidden)] Variant(u32),
}

//@ has 'repr/enum.ReprTransparentEnumPub1ZstField.html'
//@ has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // **public**
pub enum ReprTransparentEnumPub1ZstField {
    Variant {
        field: u64, // non-1-ZST field
        #[doc(hidden)]
        marker: Marker,
    },
}

//@ has 'repr/enum.ReprTransparentEnumHidden1ZstField.html'
//@ !has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // private
pub enum ReprTransparentEnumHidden1ZstField {
    Variant {
        #[doc(hidden)]
        field: u64, // non-1-ZST field
    },
}

struct Marker; // 1-ZST
