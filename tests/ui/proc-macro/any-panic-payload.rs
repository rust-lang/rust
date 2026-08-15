//! Make sure that proc-macros which panic with a payload other than
//! `String` or `&'static str` do not ICE.
//@ proc-macro: any-panic-payload.rs

extern crate any_panic_payload;

use any_panic_payload::*;

cause_panic!(); //~ ERROR proc macro panicked

#[cause_panic_attr] //~ ERROR custom attribute panicked
struct A;

#[derive(CausePanic)] //~ ERROR proc-macro derive panicked
struct B;

fn main() {}
