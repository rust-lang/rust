extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(NonAsciiIdent)]
pub fn derive_non_ascii_ident(_: TokenStream) -> TokenStream {
    "#[allow(non_ascii_idents)] const föö: () = ();".parse().unwrap()
}
