#![feature(proc_macro_quote, proc_macro_span)]

extern crate proc_macro;

use proc_macro::{Ident, Literal, Span, TokenStream, TokenTree, quote};

#[proc_macro]
pub fn bad_tup_indexing(input: TokenStream) -> TokenStream {
    let tt = input.into_iter().next().unwrap();
    let TokenTree::Literal(indexing_expr) = tt else {
        unreachable!();
    };
    quote! { (42,).$indexing_expr }
}

// Expects {IDENT, COMMA, LITERAL}
#[proc_macro]
pub fn bad_tup_struct_indexing(input: TokenStream) -> TokenStream {
    let mut input = input.into_iter();

    let ident = input.next().unwrap();
    let _comma = input.next().unwrap();
    let lit = input.next().unwrap();

    let TokenTree::Ident(ident) = ident else {
        unreachable!("id");
    };
    let TokenTree::Literal(indexing_expr) = lit else {
        unreachable!("lit");
    };

    quote! { $ident.$indexing_expr }
}
