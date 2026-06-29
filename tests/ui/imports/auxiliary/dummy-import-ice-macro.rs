extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn my_macro(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    r"
        use own::*;
        mod own {
            pub use super::submodule::*;
            pub use super::ambiguous;
        }
    "
    .parse()
    .unwrap()
}
