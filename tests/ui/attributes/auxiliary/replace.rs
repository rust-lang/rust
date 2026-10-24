//@ edition: 2018..

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn with(arg: TokenStream, _: TokenStream) -> TokenStream {
    arg
}
