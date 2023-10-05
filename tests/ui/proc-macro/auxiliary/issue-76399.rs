// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn m(_input: TokenStream) -> TokenStream {
    eprintln!("{:#?}", TokenStream::from(TokenTree::Punct(Punct::new(':', Spacing::Joint))));
    TokenStream::new()
}
