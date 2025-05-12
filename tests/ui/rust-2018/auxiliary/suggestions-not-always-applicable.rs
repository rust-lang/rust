extern crate proc_macro;

use proc_macro::*;

#[proc_macro_attribute]
pub fn foo(_attr: TokenStream, _f: TokenStream) -> TokenStream {
    "pub fn foo() -> ::Foo { ::Foo }".parse().unwrap()
}
