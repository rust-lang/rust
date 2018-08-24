// edition:2018
// aux-build:edition-kw-macro-2018.rs

#[macro_use]
extern crate edition_kw_macro_2018;

pub fn check_async() {
    let mut async = 1; //~ ERROR expected identifier, found reserved keyword `async`
    let mut r#async = 1; // OK

    r#async = consumes_async!(async); // OK
    r#async = consumes_async!(r#async); //~ ERROR no rules expected the token `r#async`
    r#async = consumes_async_raw!(async); //~ ERROR no rules expected the token `async`
    r#async = consumes_async_raw!(r#async); // OK

    if passes_ident!(async) == 1 {}
    if passes_ident!(r#async) == 1 {} // OK
    module::async(); //~ ERROR expected identifier, found reserved keyword `async`
    module::r#async(); // OK
}
