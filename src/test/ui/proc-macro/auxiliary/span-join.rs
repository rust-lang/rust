#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::{Ident, JoinError, Span, TokenStream};

#[proc_macro]
fn hygiene_differ(input: TokenStream) -> TokenStream {
    let call_site_ident = input.into_iter().next();

    let mixed_site_ident = Ident::new("other", Span::mixed_site());

    let e = call_site_ident
        .span()
        .join(mixed_site_ident.span())
        .expect_err("successfully joined despite different hygiene");

    assert_eq(e, JoinError::DifferentHygiene);

    TokenStream::new()
}
