// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::{Ident, Span, TokenStream};

#[proc_macro]
pub fn panic_in_libproc_macro(_: TokenStream) -> TokenStream {
    Ident::new("", Span::call_site());
    TokenStream::new()
}
