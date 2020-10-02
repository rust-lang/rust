// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn my_macro(_attr: TokenStream, input: TokenStream) -> TokenStream {
    println!("Called proc_macro_hack with {:?}", input);
    input
}
