#![feature(proc_macro_quote)]

extern crate proc_macro;

use std::net::Ipv4Addr;

use proc_macro::quote;

fn main() {
    let ip = Ipv4Addr::LOCALHOST;
    let _ = quote! { $ip }; //~ ERROR the trait bound `Ipv4Addr: ToTokens` is not satisfied
}
