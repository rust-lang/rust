// compile-flags: --emit=link
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![feature(repr128, proc_macro_hygiene, proc_macro_quote, box_patterns)]
#![allow(incomplete_features)]
#![allow(clippy::useless_conversion)]

extern crate proc_macro;
extern crate quote;
extern crate syn;

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::parse_macro_input;
use syn::spanned::Spanned;
use syn::token::Star;
use syn::{
    parse_quote, FnArg, ImplItem, ItemImpl, ItemTrait, Lifetime, Pat, PatIdent, PatType, Signature, TraitItem, Type,
};

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

#[proc_macro_attribute]
pub fn rename_my_lifetimes(_args: TokenStream, input: TokenStream) -> TokenStream {
    fn make_name(count: usize) -> String {
        format!("'life{}", count)
    }

    fn mut_receiver_of(sig: &mut Signature) -> Option<&mut FnArg> {
        let arg = sig.inputs.first_mut()?;
        if let FnArg::Typed(PatType { pat, .. }) = arg {
            if let Pat::Ident(PatIdent { ident, .. }) = &**pat {
                if ident == "self" {
                    return Some(arg);
                }
            }
        }
        None
    }

    let mut elided = 0;
    let mut item = parse_macro_input!(input as ItemImpl);

    // Look for methods having arbitrary self type taken by &mut ref
    for inner in &mut item.items {
        if let ImplItem::Method(method) = inner {
            if let Some(FnArg::Typed(pat_type)) = mut_receiver_of(&mut method.sig) {
                if let box Type::Reference(reference) = &mut pat_type.ty {
                    // Target only unnamed lifetimes
                    let name = match &reference.lifetime {
                        Some(lt) if lt.ident == "_" => make_name(elided),
                        None => make_name(elided),
                        _ => continue,
                    };
                    elided += 1;

                    // HACK: Syn uses `Span` from the proc_macro2 crate, and does not seem to reexport it.
                    // In order to avoid adding the dependency, get a default span from a non-existent token.
                    // A default span is needed to mark the code as coming from expansion.
                    let span = Star::default().span();

                    // Replace old lifetime with the named one
                    let lifetime = Lifetime::new(&name, span);
                    reference.lifetime = Some(parse_quote!(#lifetime));

                    // Add lifetime to the generics of the method
                    method.sig.generics.params.push(parse_quote!(#lifetime));
                }
            }
        }
    }

    TokenStream::from(quote!(#item))
}
