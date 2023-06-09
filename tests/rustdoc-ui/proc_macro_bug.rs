// regression test for failing to pass `--crate-type proc-macro` to rustdoc
// when documenting a proc macro crate https://github.com/rust-lang/rust/pull/107291

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro_derive(DeriveA)]
//~^ ERROR the `#[proc_macro_derive]` attribute is only usable with crates of the `proc-macro` crate type
pub fn a_derive(input: TokenStream) -> TokenStream {
    input
}
