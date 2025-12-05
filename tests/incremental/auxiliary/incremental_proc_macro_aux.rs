extern crate proc_macro;

use proc_macro::TokenStream;

// Add a function to shift DefIndex of registrar function
#[cfg(cfail2)]
fn foo() {}

#[proc_macro_derive(IncrementalMacro)]
pub fn derive(input: TokenStream) -> TokenStream {
    #[cfg(cfail2)]
    {
        foo();
    }

    "".parse().unwrap()
}
