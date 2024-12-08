extern crate proc_macro;
use proc_macro::{TokenStream, Ident, Span};

#[proc_macro]
pub fn panic_in_libproc_macro(_: TokenStream) -> TokenStream {
    Ident::new("", Span::call_site());
    TokenStream::new()
}
