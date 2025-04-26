extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn attr_cfg(args: TokenStream, input: TokenStream) -> TokenStream {
    let input_str = input.to_string();

    assert_eq!(input_str, "fn outer() -> u8
{
    #[cfg(foo)] fn inner() -> u8 { 1 } #[cfg(bar)] fn inner() -> u8 { 2 }
    inner()
}");

    input
}
