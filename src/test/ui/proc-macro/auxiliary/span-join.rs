#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::{Ident, JoinError, Span, TokenStream};

#[proc_macro]
fn hygiene_differ(input: TokenStream) -> TokenStream {
    let ident = input.into_iter().next();

    let mixed_site_ident = Ident::new("other", Span::mixed_site());

    let e =
        ident.join(mixed_site_ident).expect_err("successfully joined despite different hygiene");

    assert_eq(e, JoinError::DifferentHygiene);

    TokenStream::new()
}
