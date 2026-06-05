#![crate_name = "attributes_aux"]
#![feature(register_tool)]
#![register_tool(my_tool)]
#![allow(dead_code)]

#[doc = "Cross-crate struct used for reflection tests."]
#[allow(dead_code)]
#[non_exhaustive]
#[my_tool::struct_tag = "cross_crate_struct"]
pub struct CrossCrateStruct {
    #[my_tool::field_tag = "public_field"]
    pub first: u8,
    #[allow(unused_variables)]
    pub second: u16,
}

#[doc = "Cross-crate enum used for reflection tests."]
#[allow(dead_code)]
#[non_exhaustive]
#[my_tool::enum_tag = "cross_crate"]
pub enum CrossCrateEnum {
    #[allow(dead_code)]
    #[my_tool::variant_tag = "variant_a"]
    VariantA(#[my_tool::field_tag = "variant_field"] u8),
    #[my_tool::empty_variant_tag]
    VariantB,
}
