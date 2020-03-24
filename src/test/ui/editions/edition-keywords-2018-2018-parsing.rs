// edition:2018
// aux-build:edition-kw-macro-2018.rs

#![feature(async_closure)]

fn main() {}

#[macro_use]
extern crate edition_kw_macro_2018;

mod module {
    pub fn r#async() {}
}

pub fn check_async() {
    let mut async = 1; //~ ERROR expected identifier, found keyword `async`
    let mut r#async = 1; // OK

    r#async = consumes_async!(async); // OK
    r#async = consumes_async!(r#async); //~ ERROR no rules expected the token `r#async`
    r#async = consumes_async_raw!(async); //~ ERROR no rules expected the token `async`
    r#async = consumes_async_raw!(r#async); // OK

    if passes_ident!(async) == 1 {}
    if passes_ident!(r#async) == 1 {} // OK
    module::async(); //~ ERROR expected identifier, found keyword `async`
    module::r#async(); // OK

    let _recovery_witness: () = 0; //~ ERROR mismatched types
}
