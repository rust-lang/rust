// edition:2015
// aux-build:edition-kw-macro-2015.rs

#[macro_use]
extern crate edition_kw_macro_2015;

mod module {
    pub fn async() {}
}

pub fn check_async() {
    let mut async = 1; // OK
    let mut r#async = 1; // OK

    r#async = consumes_async!(async); // OK
    r#async = consumes_async!(r#async); //~ ERROR no rules expected the token `r#async`
    r#async = consumes_async_raw!(async); //~ ERROR no rules expected the token `async`
    r#async = consumes_async_raw!(r#async); // OK

    if passes_ident!(async) == 1 {} // OK
    if passes_ident!(r#async) == 1 {} // OK
    module::async(); // OK
    module::r#async(); // OK
}

fn main() {}
