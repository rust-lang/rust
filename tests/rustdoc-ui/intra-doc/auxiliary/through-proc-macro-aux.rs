// force-host
// no-prefer-dynamic
#![crate_type = "proc-macro"]
#![crate_name="some_macros"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn first(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item // This doesn't erase the spans.
}

#[proc_macro_attribute]
pub fn second(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Make a new `TokenStream` to erase the spans:
    let mut out: TokenStream = TokenStream::new();
    out.extend(item);
    out
}
