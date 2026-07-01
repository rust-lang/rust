// Regression test for <https://github.com/rust-lang/rust/issues/101531>,
// where links where to the item who's HTML page had the item linked to.

//! [`Struct::struct_field`]
//! [`Enum::Variant`]
//! [`Enum::StructVariant::field`]
//! [`Trait::AssocType`]
//! [`Trait::ASSOC_CONST`]
//! [`Trait::method`]
//! [`std::vec::Vec::push`]
//! [`std::io::ErrorKind::NotFound`]
//! [`usize::MAX`]

//@ set struct_field = "$.index[?(@.name=='struct_field')].id"
//@ set Variant = "$.index[?(@.name=='Variant')].id"
//@ set AssocType = "$.index[?(@.name=='AssocType')].id"
//@ set ASSOC_CONST = "$.index[?(@.name=='ASSOC_CONST')].id"
//@ set method = "$.index[?(@.name=='method')].id"

//@ is "$.index[?(@.name=='non_page')].links['`Struct::struct_field`']" $struct_field
//@ is "$.index[?(@.name=='non_page')].links['`Enum::Variant`']" $Variant
//@ is "$.index[?(@.name=='non_page')].links['`Trait::AssocType`']" $AssocType
//@ is "$.index[?(@.name=='non_page')].links['`Trait::ASSOC_CONST`']" $ASSOC_CONST
//@ is "$.index[?(@.name=='non_page')].links['`Trait::method`']" $method

// Regression test for <https://github.com/rust-lang/rust/issues/152511>:
// link target IDs for associated items need matching `paths` entries.
//@ has "$.paths[*].path" '["non_page", "Struct", "struct_field"]'
//@ has "$.paths[*].path" '["non_page", "Enum", "StructVariant", "field"]'
//@ has "$.paths[*].path" '["non_page", "Trait", "AssocType"]'
//@ has "$.paths[*].path" '["non_page", "Trait", "ASSOC_CONST"]'
//@ has "$.paths[*].path" '["non_page", "Trait", "method"]'
//@ has "$.paths[*].path" '["alloc", "vec", "Vec", "push"]'
//@ has "$.paths[*].path" '["core", "io", "error", "ErrorKind", "NotFound"]'
//@ has "$.paths[*].path" '["std", "usize", "MAX"]'

pub struct Struct {
    pub struct_field: i32,
}

pub enum Enum {
    Variant(),
    StructVariant { field: i32 },
}

pub trait Trait {
    const ASSOC_CONST: i32;
    type AssocType;
    fn method();
}
