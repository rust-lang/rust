//! Regression test for <https://github.com/rust-lang/rust/issues/37665>.
//! Test type error when compiling with `unpretty=mir` doesn't ICE.
//@ compile-flags: -Z unpretty=mir

use std::path::MAIN_SEPARATOR;

fn main() {
    let mut foo : String = "hello".to_string();
    foo.push(MAIN_SEPARATOR);
    println!("{}", foo);
    let x: () = 0; //~ ERROR: mismatched types
}
