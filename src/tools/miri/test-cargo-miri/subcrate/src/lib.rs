// This is a proc-macro crate.

extern crate proc_macro; // make sure proc_macro is in the sysroot

#[cfg(doctest)]
compile_error!("rustdoc should not touch me");

#[cfg(miri)]
compile_error!("Miri should not touch me");

use proc_macro::TokenStream;

#[proc_macro]
pub fn make_answer(_item: TokenStream) -> TokenStream {
    "fn answer() -> u32 { 42 }".parse().unwrap()
}
