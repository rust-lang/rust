#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::quote;

fn main() {
    quote!($(a b),*); //~ ERROR mismatched types
}
