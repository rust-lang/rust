use proc_macro2::TokenStream;

use crate::item_enum::define_config_type_on_enum;
use crate::item_struct::define_config_type_on_struct;

pub fn define_config_type(input: &syn::Item) -> TokenStream {
    match input {
        syn::Item::Struct(st) => define_config_type_on_struct(st),
        syn::Item::Enum(en) => define_config_type_on_enum(en),
        _ => panic!("Expected enum or struct"),
    }
    .unwrap()
}
