// Auxiliary proc-macro for issue #99173
// Tests that nested proc-macro calls with empty output don't cause ICE

extern crate proc_macro;

use proc_macro::TokenStream;

// This macro returns an empty TokenStream
#[proc_macro]
pub fn ignore(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}

// This macro generates code that calls the `ignore` macro
#[proc_macro]
pub fn outer_macro(_input: TokenStream) -> TokenStream {
    "nested_empty_proc_macro::ignore!(42)".parse().unwrap()
}
