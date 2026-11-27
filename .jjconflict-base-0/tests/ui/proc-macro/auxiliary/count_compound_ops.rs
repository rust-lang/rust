#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, Spacing, Literal, quote};

#[proc_macro]
pub fn count_compound_ops(input: TokenStream) -> TokenStream {
    assert_eq!(count_compound_ops_helper(quote!(++ (&&) 4@a)), 3);
    let l = Literal::u32_suffixed(count_compound_ops_helper(input));
    TokenTree::from(l).into()
}

fn count_compound_ops_helper(input: TokenStream) -> u32 {
    let mut count = 0;
    for token in input {
        match &token {
            TokenTree::Punct(tt) if tt.spacing() == Spacing::Alone => {
                count += 1;
            }
            TokenTree::Group(tt) => {
                count += count_compound_ops_helper(tt.stream());
            }
            _ => {}
        }
    }
    count
}
