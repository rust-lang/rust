// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::*;

#[proc_macro]
pub fn tokens(input: TokenStream) -> TokenStream {
    assert_nothing_joint(input);
    TokenStream::new()
}

#[proc_macro_attribute]
pub fn nothing(_: TokenStream, input: TokenStream) -> TokenStream {
    assert_nothing_joint(input);
    TokenStream::new()
}

fn assert_nothing_joint(s: TokenStream) {
    for tt in s {
        match tt {
            TokenTree::Group(g) => assert_nothing_joint(g.stream()),
            TokenTree::Punct(p) => assert_eq!(p.spacing(), Spacing::Alone),
            _ => {}
        }
    }
}
