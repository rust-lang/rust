//@ check-pass

#![feature(proc_macro_quote)]
#![feature(proc_macro_totokens)]
#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::*;

fn main() {
    let x = Ident::new("foo", Span::call_site());
    let _ = quote! {
        let $x = 199;
    };
}
