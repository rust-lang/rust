//@ no-prefer-dynamic
//@ edition: 2024

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree};

#[proc_macro_derive(Yop)]
pub fn derive_yop(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();

    while let Some(token) = iter.next() {
        if let TokenTree::Ident(ident) = token &&
            matches!(ident.to_string().as_str(), "struct" | "enum" | "union")
        {
            // Next token is the name. That's all we need!
            let Some(TokenTree::Ident(ident)) = iter.next() else { panic!() };
            return format!("impl Trait for {ident} {{}}").parse().unwrap();
        }
    }
    panic!()
}
