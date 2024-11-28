#![feature(proc_macro_quote)]

extern crate proc_macro;
use proc_macro::{TokenStream, quote};

#[proc_macro_attribute]
pub fn first_attr(_: TokenStream, input: TokenStream) -> TokenStream {
    let recollected: TokenStream = input.into_iter().collect();
    println!("First recollected: {:#?}", recollected);
    quote! {
        #[second_attr]
        $recollected
    }
}

#[proc_macro_attribute]
pub fn second_attr(_: TokenStream, input: TokenStream) -> TokenStream {
    let recollected: TokenStream = input.into_iter().collect();
    println!("Second recollected: {:#?}", recollected);
    TokenStream::new()
}
