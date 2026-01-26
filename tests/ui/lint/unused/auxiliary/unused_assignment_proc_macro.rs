#![feature(proc_macro_quote)]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro_derive(Drop)]
pub fn generate(ts: TokenStream) -> TokenStream {
    let mut ts = ts.into_iter();
    let _pub = ts.next();
    let _struct = ts.next();
    let name = ts.next().unwrap();
    let TokenTree::Group(fields) = ts.next().unwrap() else { panic!() };
    let mut fields = fields.stream().into_iter();
    let field = fields.next().unwrap();

    quote! {
        impl Drop for $name {
            fn drop(&mut self) {
                let Self { $field } = self;
            }
        }
    }
}
