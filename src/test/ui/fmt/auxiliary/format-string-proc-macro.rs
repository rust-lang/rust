// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::{Literal, Span, TokenStream, TokenTree};

#[proc_macro]
pub fn foo_with_input_span(input: TokenStream) -> TokenStream {
    let span = input.into_iter().next().unwrap().span();

    let mut lit = Literal::string("{foo}");
    lit.set_span(span);

    TokenStream::from(TokenTree::Literal(lit))
}

#[proc_macro]
pub fn err_with_input_span(input: TokenStream) -> TokenStream {
    let span = input.into_iter().next().unwrap().span();

    let mut lit = Literal::string("         }");
    lit.set_span(span);

    TokenStream::from(TokenTree::Literal(lit))
}
