// force-host
// no-prefer-dynamic
#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree};

#[proc_macro]
pub fn repro(input: TokenStream) -> TokenStream {
    for token in input {
        if let TokenTree::Literal(literal) = token {
            assert!(format!("{}", literal).contains(&"c\""), "panic on: `{}`", literal);
        }
    }
    TokenStream::new()
}
