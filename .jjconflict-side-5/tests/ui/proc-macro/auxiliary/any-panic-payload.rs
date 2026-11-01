extern crate proc_macro;

use proc_macro::TokenStream;

fn boom() -> TokenStream {
    std::panic::panic_any(42)
}

#[proc_macro]
pub fn cause_panic(_: TokenStream) -> TokenStream {
    boom()
}

#[proc_macro_attribute]
pub fn cause_panic_attr(_: TokenStream, _: TokenStream) -> TokenStream {
    boom()
}

#[proc_macro_derive(CausePanic)]
pub fn cause_panic_derive(_: TokenStream) -> TokenStream {
    boom()
}
