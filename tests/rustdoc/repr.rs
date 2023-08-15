// Regression test for <https://github.com/rust-lang/rust/issues/90435>.
// Check that we show `#[repr(transparent)]` iff the non-1-ZST field is public or at least one
// field is public in case all fields are 1-ZST fields.

// @has 'repr/struct.ReprTransparentPrivField.html'
// @!has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // private
pub struct ReprTransparentPrivField {
    field: u32, // non-1-ZST field
}

// @has 'repr/struct.ReprTransparentPriv1ZstFields.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // public
pub struct ReprTransparentPriv1ZstFields {
    marker0: Marker,
    pub main: u64, // non-1-ZST field
    marker1: Marker,
}

// @has 'repr/struct.ReprTransparentPub1ZstField.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(transparent)]'
#[repr(transparent)] // public
pub struct ReprTransparentPub1ZstField {
    marker0: Marker,
    pub marker1: Marker,
}

struct Marker; // 1-ZST
