extern crate proc_macro;

use proc_macro::*;

#[proc_macro]
pub fn neg_one(_input: TokenStream) -> TokenStream {
    TokenTree::Literal(Literal::i32_suffixed(-1)).into()
}

#[proc_macro]
pub fn neg_one_float(_input: TokenStream) -> TokenStream {
    TokenTree::Literal(Literal::f32_suffixed(-1.0)).into()
}
