//@ run-pass

#![allow(unused_assignments)]
//@ edition:2018
//@ aux-build:edition-kw-macro-2015.rs

#[macro_use]
extern crate edition_kw_macro_2015;

pub fn check_async() {
    // let mut async = 1; // ERROR, reserved
    let mut r#async = 1; // OK

    r#async = consumes_async!(async); // OK
    // r#async = consumes_async!(r#async); // ERROR, not a match
    // r#async = consumes_async_raw!(async); // ERROR, not a match
    r#async = consumes_async_raw!(r#async); // OK

    // if passes_ident!(async) == 1 {} // ERROR, reserved
    if passes_ident!(r#async) == 1 {} // OK
    // if passes_tt!(async) == 1 {} // ERROR, reserved
    if passes_tt!(r#async) == 1 {} // OK
    // one_async::async(); // ERROR, reserved
    one_async::r#async(); // OK
    // two_async::async(); // ERROR, reserved
    two_async::r#async(); // OK
}

mod one_async {
    produces_async! {} // OK
}
mod two_async {
    produces_async_raw! {} // OK
}

fn main() {}
