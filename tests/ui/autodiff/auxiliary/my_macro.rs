extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
#[macro_use]
pub fn autodiff_forward(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item // identity proc-macro
}
