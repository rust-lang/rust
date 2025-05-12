extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn assert_input(args: TokenStream, input: TokenStream) -> TokenStream {
    assert_eq!(input.to_string(), "trait Alias = Sized;");
    assert!(args.is_empty());
    TokenStream::new()
}
