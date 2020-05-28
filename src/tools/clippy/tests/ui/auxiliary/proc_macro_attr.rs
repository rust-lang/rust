// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(repr128, proc_macro_hygiene, proc_macro_quote)]
#![allow(clippy::useless_conversion)]

extern crate proc_macro;
extern crate quote;
extern crate syn;

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::parse_macro_input;
use syn::{parse_quote, ItemTrait, TraitItem};

#[proc_macro_attribute]
pub fn fake_async_trait(_args: TokenStream, input: TokenStream) -> TokenStream {
    let mut item = parse_macro_input!(input as ItemTrait);
    for inner in &mut item.items {
        if let TraitItem::Method(method) = inner {
            let sig = &method.sig;
            let block = &mut method.default;
            if let Some(block) = block {
                let brace = block.brace_token;

                let my_block = quote_spanned!( brace.span => {
                    // Should not trigger `empty_line_after_outer_attr`
                    #[crate_type = "lib"]
                    #sig #block
                    Vec::new()
                });
                *block = parse_quote!(#my_block);
            }
        }
    }
    TokenStream::from(quote!(#item))
}
