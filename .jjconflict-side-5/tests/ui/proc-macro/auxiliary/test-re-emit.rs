extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn remove_span(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // `.to_string().parse()` will lose the span of the token stream
    item.to_string().parse().unwrap()
}
