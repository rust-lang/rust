#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::{TokenStream, quote};

#[proc_macro_derive(AnotherMacro, attributes(pointee))]
pub fn derive(_input: TokenStream) -> TokenStream {
    quote! {
        const _: () = {
            const ANOTHER_MACRO_DERIVED: () = ();
        };
    }
    .into()
}

#[proc_macro_attribute]
pub fn pointee(
    _attr: proc_macro::TokenStream,
    _item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    quote! {
        const _: () = {
            const POINTEE_MACRO_ATTR_DERIVED: () = ();
        };
    }
    .into()
}

#[proc_macro_attribute]
pub fn default(
    _attr: proc_macro::TokenStream,
    _item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    quote! {
        const _: () = {
            const DEFAULT_MACRO_ATTR_DERIVED: () = ();
        };
    }
    .into()
}
