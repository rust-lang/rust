// aux-build:env.rs
// run-pass
// rustc-env: THE_CONST=1
// compile-flags: -Zunstable-options --env THE_CONST=12 --env ANOTHER=4

#![crate_name = "foo"]

extern crate env;

use env::generate_const;

generate_const!();

fn main() {
    assert_eq!(THE_CONST, 12);
    assert_eq!(ANOTHER, 1);
}
