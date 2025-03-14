extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_derive(Sample)]
pub fn sample(_: TokenStream) -> TokenStream {
    "fn bad<T: Into<U>, U>(a: T) -> U { a }".parse().unwrap()
}
