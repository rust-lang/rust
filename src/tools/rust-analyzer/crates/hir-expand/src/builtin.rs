//! Builtin macros and attributes
#[macro_use]
pub mod quote;

mod attr_macro;
mod derive_macro;
mod fn_macro;

pub use self::{
    attr_macro::{find_builtin_attr, pseudo_derive_attr_expansion, BuiltinAttrExpander},
    derive_macro::{find_builtin_derive, BuiltinDeriveExpander},
    fn_macro::{
        find_builtin_macro, include_input_to_file_id, BuiltinFnLikeExpander, EagerExpander,
    },
};
