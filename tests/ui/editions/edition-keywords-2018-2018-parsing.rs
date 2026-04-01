//@ edition:2018
//@ aux-build:edition-kw-macro-2018.rs

fn main() {}

#[macro_use]
extern crate edition_kw_macro_2018;

mod module {
    pub fn r#async() {}
}

macro_rules! local_passes_ident {
    ($i: ident) => ($i) //~ ERROR macro expansion ends with an incomplete expression
}
macro_rules! local_passes_tt {
    ($i: tt) => ($i)
}

pub fn check_async() {
    let mut async = 1; //~ ERROR expected identifier, found keyword `async`
    let mut r#async = 1; // OK

    r#async = consumes_async!(async); // OK
    r#async = consumes_async!(r#async); //~ ERROR no rules expected `r#async`
    r#async = consumes_async_raw!(async); //~ ERROR no rules expected keyword `async`
    r#async = consumes_async_raw!(r#async); // OK

    if passes_ident!(async) == 1 {} // FIXME: Edition hygiene bug, async here is 2018 and reserved
    if passes_ident!(r#async) == 1 {} // OK
    if passes_tt!(async) == 1 {} //~ ERROR macro expansion ends with an incomplete expression
    if passes_tt!(r#async) == 1 {} // OK
    if local_passes_ident!(async) == 1 {} // Error reported above in the macro
    if local_passes_ident!(r#async) == 1 {} // OK
    if local_passes_tt!(async) == 1 {} //~ ERROR macro expansion ends with an incomplete expression
    if local_passes_tt!(r#async) == 1 {} // OK
    module::async(); //~ ERROR expected identifier, found keyword `async`
    module::r#async(); // OK
}

//~? ERROR macro expansion ends with an incomplete expression
