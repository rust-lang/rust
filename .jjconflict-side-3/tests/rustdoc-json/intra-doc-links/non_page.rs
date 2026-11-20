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
//@ jq_set crate = '.index[] | select(.name == "non_page")'
//@ jq_set struct_field_link = '$crate.links["`Struct::struct_field`"]'
//@ jq_set variant_field_link = '$crate.links["`Enum::StructVariant::field`"]'
//@ jq_set assoc_type_link = '$crate.links["`Trait::AssocType`"]'
//@ jq_set assoc_const_link = '$crate.links["`Trait::ASSOC_CONST`"]'
//@ jq_set method_link = '$crate.links["`Trait::method`"]'
//@ jq_set vec_push_link = '$crate.links["`std::vec::Vec::push`"]'
//@ jq_set error_kind_link = '$crate.links["`std::io::ErrorKind::NotFound`"]'
//@ jq_set usize_max_link = '$crate.links["`usize::MAX`"]'
//@ jq_is '.paths["\($struct_field_link)"].path' '["non_page", "Struct", "struct_field"]'
//@ jq_is '.paths["\($variant_field_link)"].path' '["non_page", "Enum", "StructVariant", "field"]'
//@ jq_is '.paths["\($assoc_type_link)"].path' '["non_page", "Trait", "AssocType"]'
//@ jq_is '.paths["\($assoc_const_link)"].path' '["non_page", "Trait", "ASSOC_CONST"]'
//@ jq_is '.paths["\($method_link)"].path' '["non_page", "Trait", "method"]'
//@ jq_is '.paths["\($vec_push_link)"].path' '["alloc", "vec", "Vec", "push"]'
//@ jq_is '.paths["\($error_kind_link)"].path' '["core", "io", "error", "ErrorKind", "NotFound"]'
//@ jq_is '.paths["\($usize_max_link)"].path' '["std", "usize", "MAX"]'

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
