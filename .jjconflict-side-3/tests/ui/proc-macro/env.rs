//@ proc-macro: env.rs
//@ run-pass
//@ rustc-env: THE_CONST=1
//@ ignore-backends: gcc

#![crate_name = "foo"]

extern crate env;

use env::generate_const;

generate_const!();

fn main() {
    assert_eq!(THE_CONST, 1);
    assert_eq!(ANOTHER, 2); // not found, see env::generate_const
}
