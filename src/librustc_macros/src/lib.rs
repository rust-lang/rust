#![feature(proc_macro_hygiene)]
#![deny(rust_2018_idioms)]

#![recursion_limit="128"]

extern crate proc_macro;

use synstructure::decl_derive;

use proc_macro::TokenStream;

mod hash_stable;
mod query;
mod symbols;

#[proc_macro]
pub fn rustc_queries(input: TokenStream) -> TokenStream {
    query::rustc_queries(input)
}

#[proc_macro]
pub fn symbols(input: TokenStream) -> TokenStream {
    symbols::symbols(input)
}

decl_derive!([HashStable, attributes(stable_hasher)] => hash_stable::hash_stable_derive);
