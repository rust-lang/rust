extern crate proc_macro;

use proc_macro::TokenStream;
use std::str::FromStr;

#[proc_macro]
pub fn call_site_path(_: TokenStream) -> TokenStream {
    TokenStream::from_str(
        "pub fn proc_call_site() -> private_macros::Hidden { loop {} }",
    )
    .unwrap()
}
