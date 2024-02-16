//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn bad_input(input: String) -> TokenStream {
    //~^ ERROR function-like proc macro has incorrect signature
    ::proc_macro::TokenStream::new()
}

#[proc_macro]
pub fn bad_output(input: TokenStream) -> String {
    //~^ ERROR function-like proc macro has incorrect signature
    String::from("blah")
}

#[proc_macro]
pub fn bad_everything(input: String) -> String {
    //~^ ERROR function-like proc macro has incorrect signature
    input
}

#[proc_macro]
pub fn too_many(a: TokenStream, b: TokenStream, c: String) -> TokenStream {
    //~^ ERROR function-like proc macro has incorrect signature
}
