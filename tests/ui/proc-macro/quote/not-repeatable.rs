#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::quote;

struct Ipv4Addr;

fn main() {
    let ip = Ipv4Addr;
    let _ = quote! { $($ip)* };
    //~^ ERROR the method `quote_into_iter` exists for struct `Ipv4Addr`, but its trait bounds were not satisfied
    //~| ERROR type annotations needed
}
