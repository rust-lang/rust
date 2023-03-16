// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn bad_input(input: String) -> TokenStream {
    //~^ ERROR attribute proc macro has incorrect signature
    ::proc_macro::TokenStream::new()
}

#[proc_macro_attribute]
pub fn bad_output(input: TokenStream) -> String {
    //~^ ERROR attribute proc macro has incorrect signature
    String::from("blah")
}

#[proc_macro_attribute]
pub fn bad_everything(input: String) -> String {
    //~^ ERROR attribute proc macro has incorrect signature
    input
}

#[proc_macro_attribute]
pub fn too_many(a: TokenStream, b: TokenStream, c: String) -> TokenStream {
    //~^ ERROR attribute proc macro has incorrect signature
}
