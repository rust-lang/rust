// Regression test for <https://github.com/rust-lang/rust/issues/101531>,
// where links where to the item who's HTML page had the item linked to.

//! [`Struct::struct_field`]
//! [`Enum::Variant`]
//! [`Trait::AssocType`]
//! [`Trait::ASSOC_CONST`]
//! [`Trait::method`]

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

pub struct Struct {
    pub struct_field: i32,
}

pub enum Enum {
    Variant(),
}

pub trait Trait {
    const ASSOC_CONST: i32;
    type AssocType;
    fn method();
}
