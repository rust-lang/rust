// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --edition=2015
// aux-build:edition-kw-macro-2015.rs

#![feature(raw_identifiers)]

#[macro_use]
extern crate edition_kw_macro_2015;

// `async`
pub fn check_async() {
    let mut async = 1; // OK
    let mut r#async = 1; // OK

    r#async = consumes_async!(async); // OK
    // r#async = consumes_async!(r#async); // ERROR, not a match
    // r#async = consumes_async_raw!(async); // ERROR, not a match
    r#async = consumes_async_raw!(r#async); // OK

    if passes_ident!(async) == 1 {} // OK
    if passes_ident!(r#async) == 1 {} // OK
    one_async::async(); // OK
    one_async::r#async(); // OK
    two_async::async(); // OK
    two_async::r#async(); // OK
}

mod one_async {
    produces_async! {} // OK
}
mod two_async {
    produces_async_raw! {} // OK
}

// `proc`
pub fn check_proc() {
    // let mut proc = 1; // ERROR, reserved
    let mut r#proc = 1; // OK

    r#proc = consumes_proc!(proc); // OK
    // r#proc = consumes_proc!(r#proc); // ERROR, not a match
    // r#proc = consumes_proc_raw!(proc); // ERROR, not a match
    r#proc = consumes_proc_raw!(r#proc); // OK

    // if passes_ident!(proc) == 1 {} // ERROR, reserved
    if passes_ident!(r#proc) == 1 {} // OK
    // one_proc::proc(); // ERROR, reserved
    // one_proc::r#proc(); // ERROR, unresolved name
    // two_proc::proc(); // ERROR, reserved
    two_proc::r#proc(); // OK
}

mod one_proc {
    // produces_proc! {} // ERROR, reserved
}
mod two_proc {
    produces_proc_raw! {} // OK
}

fn main() {}
