#![crate_type = "proc-macro"]
#![deny(ambiguous_derive_helpers)]

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(Trait, attributes(ignore))] //~ ERROR there exists a built-in attribute with the same name
//~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
pub fn deriving(input: TokenStream) -> TokenStream {
    TokenStream::new()
}
