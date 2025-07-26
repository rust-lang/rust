#![feature(proc_macro_hygiene, proc_macro_quote, box_patterns)]
#![allow(clippy::useless_conversion, clippy::uninlined_format_args)]

extern crate proc_macro;
extern crate quote;
extern crate syn;

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::token::Star;
use syn::{
    FnArg, ImplItem, ItemFn, ItemImpl, ItemStruct, ItemTrait, Lifetime, Pat, PatIdent, PatType, Signature, TraitItem,
    Type, Visibility, parse_macro_input, parse_quote,
};

#[proc_macro_attribute]
pub fn dummy(_args: TokenStream, input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_attribute]
pub fn fake_async_trait(_args: TokenStream, input: TokenStream) -> TokenStream {
    let mut item = parse_macro_input!(input as ItemTrait);
    for inner in &mut item.items {
        if let TraitItem::Fn(method) = inner {
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
        if let FnArg::Typed(PatType { pat, .. }) = arg
            && let Pat::Ident(PatIdent { ident, .. }) = &**pat
            && ident == "self"
        {
            Some(arg)
        } else {
            None
        }
    }

    let mut elided = 0;
    let mut item = parse_macro_input!(input as ItemImpl);

    // Look for methods having arbitrary self type taken by &mut ref
    for inner in &mut item.items {
        if let ImplItem::Fn(method) = inner
            && let Some(FnArg::Typed(pat_type)) = mut_receiver_of(&mut method.sig)
            && let box Type::Reference(reference) = &mut pat_type.ty
        {
            // Target only unnamed lifetimes
            let name = match &reference.lifetime {
                Some(lt) if lt.ident == "_" => make_name(elided),
                None => make_name(elided),
                _ => continue,
            };
            elided += 1;

            // HACK: Syn uses `Span` from the proc_macro2 crate, and does not seem to reexport it.
            // In order to avoid adding the dependency, get a default span from a nonexistent token.
            // A default span is needed to mark the code as coming from expansion.
            let span = Star::default().span();

            // Replace old lifetime with the named one
            let lifetime = Lifetime::new(&name, span);
            reference.lifetime = Some(parse_quote!(#lifetime));

            // Add lifetime to the generics of the method
            method.sig.generics.params.push(parse_quote!(#lifetime));
        }
    }

    TokenStream::from(quote!(#item))
}

#[proc_macro_attribute]
pub fn fake_main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut item = parse_macro_input!(item as ItemFn);
    let span = item.block.brace_token.span;

    item.sig.asyncness = None;

    let crate_name = quote! { fake_crate };
    let block = item.block;
    item.block = syn::parse_quote_spanned! {
        span =>
        {
            #crate_name::block_on(async {
                #block
            })
        }
    };

    quote! {
        mod #crate_name {
            pub fn block_on<F: ::std::future::Future>(_fut: F) {}
        }

        #item
    }
    .into()
}

#[proc_macro_attribute]
pub fn fake_desugar_await(_args: TokenStream, input: TokenStream) -> TokenStream {
    let mut async_fn = parse_macro_input!(input as syn::ItemFn);

    for stmt in &mut async_fn.block.stmts {
        if let syn::Stmt::Expr(syn::Expr::Match(syn::ExprMatch { expr: scrutinee, .. }), _) = stmt
            && let syn::Expr::Await(syn::ExprAwait { base, await_token, .. }) = scrutinee.as_mut()
        {
            let blc = quote_spanned!( await_token.span => {
                #[allow(clippy::let_and_return)]
                let __pinned = #base;
                __pinned
            });
            *scrutinee = parse_quote!(#blc);
        }
    }

    quote!(#async_fn).into()
}

#[proc_macro_attribute]
pub fn rewrite_struct(_args: TokenStream, input: TokenStream) -> TokenStream {
    let mut item_struct = parse_macro_input!(input as syn::ItemStruct);
    // remove struct attributes including doc comments.
    item_struct.attrs = vec![];
    if let Visibility::Public(token) = item_struct.vis {
        // set vis to `pub(crate)` to trigger `missing_docs_in_private_items` lint.
        let new_vis: Visibility = syn::parse_quote_spanned!(token.span() => pub(crate));
        item_struct.vis = new_vis;
    }
    if let syn::Fields::Named(fields) = &mut item_struct.fields {
        for field in &mut fields.named {
            // remove all attributes from fields as well.
            field.attrs = vec![];
        }
    }

    quote!(#item_struct).into()
}

#[proc_macro_attribute]
pub fn with_empty_docs(_attr: TokenStream, input: TokenStream) -> TokenStream {
    let item = parse_macro_input!(input as syn::Item);
    let attrs: Vec<syn::Attribute> = vec![];
    let doc_comment = "";
    quote! {
        #(#attrs)*
        #[doc = #doc_comment]
        #item
    }
    .into()
}

#[proc_macro_attribute]
pub fn duplicated_attr(_attr: TokenStream, input: TokenStream) -> TokenStream {
    let item = parse_macro_input!(input as syn::Item);
    let attrs: Vec<syn::Attribute> = vec![];
    quote! {
        #(#attrs)*
        #[allow(unused)]
        #[allow(unused)]
        #[allow(unused)]
        #item
    }
    .into()
}
