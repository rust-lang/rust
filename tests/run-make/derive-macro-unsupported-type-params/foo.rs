#![crate_type = "proc-macro"]
#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, quote};

#[proc_macro_derive(A)]
pub fn derive(item: TokenStream) -> TokenStream {
    let mut tokens = item.into_iter();
    let _enum = tokens.next();
    let name = tokens.next().unwrap();
    let _ = tokens.next().unwrap();
    let _ = tokens.next().unwrap();
    let _ = tokens.next().unwrap();
    let TokenTree::Group(group) = tokens.next().unwrap() else { panic!() };
    let mut group = group.stream().into_iter();
    let variant = group.next().unwrap();
    let TokenTree::Group(args) = group.next().unwrap() else { panic!() };
    let arg = args.stream().into_iter().next().unwrap();
    let tokens = quote! {
        trait X {}
        #[automatically_derived]
        impl X for $name {}

        #[automatically_derived]
        impl $name {
            fn foo(&self) {
                if let Self :: $variant(val) = self {
                    let _: $arg = val;
                }
            }
        }

    };
    tokens
}
