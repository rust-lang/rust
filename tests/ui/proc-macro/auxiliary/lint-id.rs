#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use proc_macro::{LintId, TokenStream};

#[proc_macro_lint]
pub static ambiguous_thing: LintId;

#[proc_macro]
pub fn print_lintid_to_stderr(_input: TokenStream) -> TokenStream {
    eprintln!("printed by macro: {:?}", crate::ambiguous_thing);
    TokenStream::new()
}
