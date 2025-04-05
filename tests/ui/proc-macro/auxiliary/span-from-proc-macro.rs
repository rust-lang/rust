#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::{quote, TokenStream};

macro_rules! expand_to_quote {
    () => {
        quote! {
            let bang_error: bool = 25;
        }
    }
}

#[proc_macro]
pub fn error_from_bang(_input: TokenStream) -> TokenStream {
    expand_to_quote!()
}

#[proc_macro_attribute]
pub fn error_from_attribute(_args: TokenStream, _input: TokenStream) -> TokenStream {
    quote! {
        struct AttributeError {
            field: MissingType
        }
    }
}

#[proc_macro_derive(ErrorFromDerive)]
pub fn error_from_derive(_input: TokenStream) -> TokenStream {
    quote! {
        enum DeriveError {
            Variant(OtherMissingType)
        }
    }
}
