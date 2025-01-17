extern crate proc_macro;
use proc_macro::*;

#[proc_macro_derive(EthabiContract, attributes(ethabi_contract_options))]
pub fn ethabi_derive(input: TokenStream) -> TokenStream {
    Default::default()
}
