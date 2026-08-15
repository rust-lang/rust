#![feature(proc_macro_quote)]

extern crate proc_macro;
use proc_macro::{TokenStream, quote};

#[proc_macro]
pub fn make_it(input: TokenStream) -> TokenStream {
    // `quote!` applies def-site hygiene
    quote! {
        trait Foo {
            fn my_fn(&self) {}
        }

        impl<T> Foo for T {}
        "a".my_fn();
    }
}
