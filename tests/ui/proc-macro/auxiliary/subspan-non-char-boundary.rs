#![feature(proc_macro_span)]

extern crate proc_macro;

use proc_macro::{Literal, TokenStream, TokenTree};

#[proc_macro]
pub fn check_non_char_boundary(input: TokenStream) -> TokenStream {
    let TokenTree::Literal(lit) = input.into_iter().next().unwrap() else {
        panic!("expected a string literal");
    };
    // Byte 2 is inside '🦀'; subspan should reject it.
    assert!(lit.subspan(2..3).is_none(), "bad subspan");
    TokenStream::new()
}
