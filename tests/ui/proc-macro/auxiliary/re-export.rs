extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn cause_ice(_: TokenStream) -> TokenStream {
    "
        enum IceCause {
            Variant,
        }

        pub use IceCause::Variant;
    ".parse().unwrap()
}
