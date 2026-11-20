extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn make_edition_macro(_input: TokenStream) -> TokenStream {
    "macro_rules! edition {
        ($_:expr) => {
            2024
        };
        (const {}) => {
            2021
        };
    }
    "
    .parse()
    .unwrap()
}

#[proc_macro]
pub fn make_nested_edition_macro(_input: TokenStream) -> TokenStream {
    "macro_rules! make_inner {
        () => {
            macro_rules! edition_inner {
                ($_:expr) => {
                    2024
                };
                (const {}) => {
                    2021
                };
            }
        };
    }
    "
    .parse()
    .unwrap()
}
