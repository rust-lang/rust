//@ edition:2018

#![feature(proc_macro_def_site)]

extern crate proc_macro;
extern crate make_macro;
use proc_macro::{TokenStream, Span};

make_macro::make_it!(print_def_site);

#[proc_macro]
pub fn dummy(input: TokenStream) -> TokenStream { input }
